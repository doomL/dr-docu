from util import *
from streamlit_option_menu import option_menu
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Dr. Docu", page_icon="dr-docu-icon.ico", layout="centered")

# --- SETUP SESSION STATE VARIABLES ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = False
if "response" not in st.session_state:
    st.session_state.response = None
if "prompt_activation" not in st.session_state:
    st.session_state.prompt_activation = False
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if "prompt" not in st.session_state:
    st.session_state.prompt = False

load_dotenv()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header('Info')
groq_api_key = sidebar_api_key_configuration()
model = sidebar_groq_model_selection()

# --- MAIN PAGE CONFIGURATION ---

# st.title("Dr. Docu")+st.image("dr-docu.png", width=256,align="center")
# st.write("*Parla Direttamente con i tuoi Documenti! :books:*")
# st.write(':blue[***Powered by Groq AI Inference Technology***]')

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image("dr-docu.png", width=230)

with col2:
    st.title("Dr. Docu")
    st.write("*Parla Direttamente con i tuoi Documenti! :books:*")
    st.write(':blue[***Powered by Groq AI Inference Technology***]')
    st.write("")
    st.write("Benvenuto in Dr. Docu! Questa app ti consente di chattare con i tuoi documenti PDF. Carica i tuoi documenti, fai una domanda e ottieni una risposta accurata basata sul contenuto dei tuoi documenti. Prova subito!")

# Slider for choosing the length of history
history_length = st.sidebar.slider(
    "Select History Length",
    min_value=1,
    max_value=20,
    value=5,  # Default value
    step=1
)
    
# ---- NAVIGATION MENU -----
selected = option_menu(
    menu_title=None,
    options=["Dr. Docu", "Riferimenti", "About"],
    icons=["stars", "bi-file-text-fill", "info-circle"],  # https://icons.getbootstrap.com
    orientation="horizontal",
)

llm = ChatGroq(groq_api_key=groq_api_key, model_name=model, max_tokens=8000)

prompt = ChatPromptTemplate.from_template(
    """
    Rispondi alla domanda basandoti solo sul contesto fornito. Se la domanda non è presente nel contesto, non cercare di rispondere
    e indica che la domanda è fuori contesto o qualcosa di simile.
    Per favore, fornisci la risposta più accurata possibile in base alla domanda.
    <contesto>
    {context}
    Domande: {input}
    """
)
# ----- SETUP Doc Chat MENU ------
if selected == "Dr. Docu":
    st.subheader("Carica File PDF")
    pdf_docs = st.file_uploader("Carica I tuoi File PDF", type=['pdf'], accept_multiple_files=True,
                                disabled=not st.session_state.prompt_activation, label_visibility='collapsed')
    process = st.button("Process", type="primary", key="process", disabled=not pdf_docs)

    if process:
        with st.spinner("Elaborando ..."):
            st.session_state.vector_store = create_vectorstore(pdf_docs)
            st.session_state.prompt = True
            st.success('Processo completato con successo!')

    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Ciao! Come posso aiutarti oggi?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    container = st.container(border=True)
    if question := st.chat_input(placeholder='Inserisci la tua domanda qui...', key="question", disabled=not st.session_state.prompt):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner('Elaborando...'):
            promptH = generate_prompt_from_history(st.session_state.message_history,history_length)
            # Add the current question to the prompt
            promptH += f"user: {question}\n"
            print (promptH)

            st.session_state.response = get_llm_response(llm, prompt, promptH)
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.response['answer']})
            st.chat_message("assistant").write(st.session_state.response['answer'])
            update_message_history(question, st.session_state.response['answer'])

# ----- SETUP REFERENCE MENU ------
if selected == "Riferimenti":
    st.title("Riferimenti & Contesto")
    if st.session_state.response is not None:
        for i, doc in enumerate(st.session_state.response["context"]):
            with st.expander(f'Riferimento # {i + 1}'):
                st.write(doc.page_content)

# ----- SETUP ABOUT MENU ------
if selected == "About":
    with st.expander("Informazioni su questa App"):
        st.markdown(''' Questa app ti consente di chattare con i tuoi documenti PDF. Ha le seguenti funzionalità:

    - Consente di chattare con più documenti PDF
    - Supporto della tecnologia di inferenza Groq AI
    - Visualizza il contesto della risposta e il riferimento del documento

        ''')
    with st.expander("Quali modelli di linguaggio di grandi dimensioni sono supportati da questa app?"):
        # Fetch available models
        models = get_models()
        # Create a list of supported models
        supported_models = [f"- Modelli di chat -- {model['id']}" for model in models]
        models_list = "\n".join(supported_models)
        st.markdown(f''' Questa app supporta i seguenti LLM come supportato da Groq:

    {models_list}
        ''')

    with st.expander("Quale libreria viene utilizzata per il vectorstore?"):
        st.markdown('''
            Questa app utilizza FAISS (Facebook AI Similarity Search) per la ricerca di similarità AI e per il vectorstore.

            ### Cos'è FAISS?
            FAISS è una libreria sviluppata da Facebook AI Research che consente una ricerca efficiente di somiglianze tra vettori. È particolarmente utile per applicazioni che richiedono la ricerca di elementi simili in grandi dataset.

            ### Come viene utilizzato FAISS nel progetto?
            In questo progetto, FAISS viene utilizzato per creare un vectorstore dai documenti PDF. I documenti vengono convertiti in embeddings, che sono rappresentazioni numeriche dei testi. Questi embeddings vengono poi indicizzati utilizzando FAISS, permettendo una ricerca rapida ed efficiente di documenti simili.

            ### Vantaggi di utilizzare FAISS
            - **Efficienza**: FAISS è ottimizzato per la ricerca di similarità ad alta velocità, anche su dataset di grandi dimensioni.
            - **Scalabilità**: Può gestire milioni di vettori, rendendolo adatto per applicazioni su larga scala.
            - **Flessibilità**: Supporta diverse metriche di distanza e può essere configurato per bilanciare tra velocità e precisione.

            ### Esempio di utilizzo di FAISS
            ```python
            import faiss
            import numpy as np

            # Supponiamo di avere una lista di embeddings
            embeddings = np.array([...])  # Array di embeddings

            # Creiamo un indice FAISS
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)

            # Aggiungiamo gli embeddings all'indice
            index.add(embeddings)

            # Eseguiamo una ricerca di similarità
            query_embedding = np.array([...])  # Embedding della query
            D, I = index.search(query_embedding, k=5)  # Trova i 5 embeddings più simili

            print(I)  # Indici degli embeddings più simili
            print(D)  # Distanze degli embeddings più simili
            ```
            ''')

    with st.expander("Come funziona il modello di chat?"):
        st.markdown(''' Questa app utilizza il modello di chat Groq AI per generare risposte accurate alle domande.
        ''')

    with st.expander("Come posso utilizzare questa app?"):
        st.markdown(''' Per utilizzare questa app, è necessario caricare i documenti PDF e fare clic su Process.
        Successivamente, è possibile inserire la domanda e ottenere la risposta.
        ''')
