# Dr. Docu
<p align="center">
  <img src="dr-docu.png" alt="Doctor Docu" title="a title" />
</p>

Questo strumento consente di caricare documenti PDF, creare un archivio di vettori (vectorstore) utilizzando embeddings testuali, e ottenere risposte da un modello LLM (Large Language Model) basato sui contenuti dei PDF forniti.

## Requisiti

- **Python 3.x**
- **pip** - Il gestore di pacchetti Python.
- **Ambiente virtuale** - Consigliato per isolare le dipendenze del progetto.

## Installazione

Segui questi passaggi per configurare l'ambiente di sviluppo:

1. **Clona il repository:**
    ```bash
    git clone https://github.com/dooml/dr-docu.git
    cd dr-docu
    ```

2. **Crea e attiva un ambiente virtuale:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Su Windows, usa `venv\Scripts\activate`
    ```

3. **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Configura l'API key per Groq:**
    - Registrati su [Groq](https://www.groq.com) e genera una nuova API key.
    - Esporta l'API key come variabile di sistema:
        ```bash
        export GROQ_API_KEY=la-tua-api-key
        ```
    - Oppure Impostala all'interno del .env
    <br>

5. **Scarica e installa Ollama:**
    - Segui le istruzioni su [Ollama](https://www.ollama.com) per scaricare e installare il software.
    - Dopo l'installazione, scarica il modello di embedding eseguendo:
        ```bash
        ollama pull nomic-embed-text
        ```
    - Questo modello verrà utilizzato per generare embeddings testuali che rappresentano i contenuti dei PDF.
<br>   

6. **Esegui Il Programma**
 ```bash
   streamlit run main.py
   ```

## Esempio Codice
Ecco un esempio di utilizzo del progetto:
```
import os
from util import create_vectorstore, get_llm_response

# Carica la API key da Groq
groq_api_key = os.getenv("GROQ_API_KEY")

# Supponiamo di avere una lista di documenti PDF
pdf_docs = ["document1.pdf", "document2.pdf"]

# Crea il vectorstore utilizzando i PDF
vectorstore = create_vectorstore(pdf_docs)

# Inizializza il tuo modello LLM (esempio)
llm = ...  # Codice per inizializzare il modello

# Definisci il prompt e la domanda
prompt = "Inserisci il prompt qui"
question = "Inserisci la domanda qui"

# Ottieni una risposta dal modello LLM
response = get_llm_response(llm, prompt, question)

# Stampa la risposta ottenuta
print(response)
```

## Librerie Utilizzate

Il progetto utilizza una serie di librerie Python che facilitano l'elaborazione del linguaggio naturale, la gestione dei file PDF e l'integrazione con servizi di terze parti. Di seguito una descrizione delle principali librerie utilizzate:

- **pypdf:**  
  Una libreria Python leggera per leggere e manipolare file PDF. È utilizzata per estrarre testo, immagini e metadati dai PDF, facilitando la creazione di archivi di contenuti testuali che possono essere elaborati successivamente.

- **langchain:**  
  Una potente libreria per costruire catene di elaborazione del linguaggio naturale (NLP). LangChain semplifica l'orchestrazione di modelli di linguaggio, agenti e strumenti esterni, permettendo la costruzione di applicazioni complesse come chatbots e sistemi di Q&A basati sui documenti.

- **langchain-core:**  
  Il nucleo della libreria LangChain, fornisce le basi per creare e gestire catene di trasformazione dei dati e logica applicativa per l'NLP. Include moduli per la gestione delle pipeline, il routing delle domande e l'integrazione con diverse API.

- **langchain-groq:**  
  Un'estensione di LangChain per l'integrazione con Groq, una piattaforma per l'elaborazione di modelli di linguaggio su hardware dedicato. Questo modulo permette di sfruttare la potenza di Groq per eseguire modelli LLM con prestazioni ottimizzate.

- **langchain-community:**  
  Una raccolta di moduli e componenti aggiuntivi sviluppati dalla comunità per espandere le funzionalità di LangChain. Questi componenti includono integrazioni con altre librerie e piattaforme, nonché nuovi strumenti per l'NLP.

- **streamlit:**  
  Una libreria Python che consente di creare applicazioni web interattive per la scienza dei dati e il machine learning con poche righe di codice. Streamlit viene utilizzata per costruire interfacce utente che permettono agli utenti di caricare documenti, configurare modelli e visualizzare i risultati.

- **streamlit-option-menu:**  
  Un componente aggiuntivo per Streamlit che permette di creare menu di opzioni personalizzati. Questo migliora l'esperienza utente permettendo di navigare facilmente tra diverse sezioni dell'applicazione.

- **python-dotenv:**  
  Una libreria che semplifica la gestione delle variabili di ambiente, caricandole da un file `.env`. Questo è particolarmente utile per mantenere separati i dati sensibili (come le API keys) dal codice sorgente.

- **boto3:**  
  L'SDK AWS per Python, utilizzato per interagire con i servizi AWS come S3, EC2, e DynamoDB. In questo progetto, Boto3 potrebbe essere utilizzato per archiviare e recuperare documenti o risultati di elaborazione da S3.

- **faiss-cpu:**  
  Una libreria sviluppata da Facebook AI Research (FAISS) per la ricerca efficiente di somiglianze tra vettori. Faiss è particolarmente utile per creare e interrogare archivi di vettori (vectorstore), permettendo di cercare documenti simili o rilevanti rispetto a una query.

- **gpt4all:**  
  Una libreria per interagire con il modello GPT-4, fornendo un'interfaccia semplice per ottenere risposte da un modello di linguaggio avanzato. Viene utilizzata per generare risposte basate sui contenuti dei PDF analizzati.

## Conclusione

In questo documento abbiamo visto come utilizzare lo strumento per caricare documenti PDF, creare un archivio di vettori utilizzando embeddings testuali e ottenere risposte da un modello LLM basato sui contenuti dei PDF forniti. Abbiamo anche esaminato i requisiti di installazione, i passaggi per configurare l'ambiente di sviluppo e le librerie utilizzate nel progetto.

È importante notare che questo è solo un esempio di utilizzo e che è possibile personalizzare il codice per adattarlo alle proprie esigenze. Si consiglia di consultare la documentazione delle librerie utilizzate per ulteriori informazioni e dettagli.

Speriamo che questo documento ti sia stato utile per iniziare con il progetto. Se hai domande o hai bisogno di ulteriori informazioni, non esitare a chiedere.

Buon lavoro!
