import streamlit as st

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("Machine Translation using Transformers")
st.write(
"""
Transformers are models that have revolutionized text translation. 
They use a mechanism called self-attention to process and understand the context of words in a sequence. 
Unlike traditional models, transformers analyze entire sentences at once,
allowing them to capture relationships between distant words."""
)


st.subheader("Dataset")
st.write("""
Dataset from Wikidepia called IndoParaCrawl. 
IndoParaCrawl is ParaCrawl v7.1 dataset bulk-translated to Indonesian using Google Translate.
https://huggingface.co/datasets/Wikidepia/IndoParaCrawl
"""
)

data = list(
    [
        {
            "en": "The Red Hat Enterprise Linux System Administra...",
            "id": "Panduan Administrasi Sistem Linux Perusahaan R...",
        },
        {
            "en": "Both men and women need to do resistance and p...",
            "id": "Baik pria maupun wanita perlu melakukan pelati...",
        },
        {
            "en": "If you are unsure, you could call your doctor'...",
            "id": "Jika Anda tidak yakin, Anda dapat menghubungi ...",
        },
        {
            "en": "A suspicious object was thrown out into the co...",
            "id": "Sebuah benda mencurigakan dilempar ke koridor ...",
        },
        {
            "en": "About B1443: 4-5mm pearl magnetic bracelet",
            "id": "Tentang B1443: Gelang magnet mutiara 4-5mm",
        },
    ]
)

st.dataframe(data)


st.subheader("Preprocessing")

st.write("Adding START and STOP token.")
st.code("""
df["id"] = df["id"].apply(lambda x:  "[start] " + x + " [end]")
""")


st.write("Preprocessing data to convert string into vector.")
st.code("""
strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
strip_chars = strip_chars.replace(",", "")
strip_chars = strip_chars.replace(".", "")
strip_chars = strip_chars.replace("!", "")
strip_chars = strip_chars.replace("?", "")

vocab_size = 50_000
sequence_length = 20
batch_size = 64

@keras.saving.register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

idn_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization, # must same otherwise it not appear

)
eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize=custom_standardization,
)
""")

st.write("The dataset needs to be properly formatted to be compatible with transformers.")
st.code("""
def format_dataset(eng, idn):
    idn = idn_vectorization(idn)
    eng = eng_vectorization(eng)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": idn[:, :-1],
        },
        idn[:, 1:],
    )
""")

st.subheader("Model")
st.write("Model summary:")
st.image("https://i0.wp.com/i.postimg.cc/Bn7QmpQS/1-43lg-CTy-M5c-TTABj-C2-VEHd-A.png?resize=579%2C800&ssl=1")

st.write("Positional Embedding")
st.code("""
@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
""")

st.write("Encoder")
st.code("""
@keras.saving.register_keras_serializable()
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
""")
st.write("Decoder")
st.code("""
@keras.saving.register_keras_serializable()
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
            padding_mask = ops.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)
""")
st.write("Hyperparameter Tuning")
st.code("""
embed_dim_arr = [256, 512]
latent_dim_arr = [1024, 2048]
num_heads_arr = [8, 16]


for embed_dim in embed_dim_arr:
    for latent_dim in latent_dim_arr:
        for num_heads in num_heads_arr:
            encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
            x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
            encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
            encoder = keras.Model(encoder_inputs, encoder_outputs)
            
            decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
            encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
            x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
            x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
            x = layers.Dropout(0.5)(x)
            decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
            decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)
            
            decoder_outputs = decoder([decoder_inputs, encoder_outputs])
            transformer = keras.Model(
                [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
            )
            epochs = 30 
            transformer.compile(
                "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )
            transformer.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=2)
            transformer.evaluate(test_ds)
""")

st.subheader("Evaluation")
st.write("Model evaluation using accuracy score.")


st.subheader("Inference")
st.write("Function to generate translation from Indonesia to English.")
st.code("""
def translate(ind_text):
    idn_vec = idn_vectorization([ind_text])
    eng_start = eng_vectorization(['[start]'])
        
    decoded_sentence = []
    original = []
    max_length = 20

    for i in range(max_length):
        predictions = transformer.predict({
            'encoder_inputs': idn_vec,
            'decoder_inputs': eng_start
        }, verbose=0)
        predicted_id = np.argmax(predictions[0, i, :])
        original.append(predicted_id)
        predicted_word  = eng_index_lookup[predicted_id]
        if predicted_word == '[end]':
            break
        decoded_sentence.append(predicted_word)
        eng_start = eng_vectorization([' '.join(['[start]'] + decoded_sentence)])

    return ' '.join(decoded_sentence)
""")