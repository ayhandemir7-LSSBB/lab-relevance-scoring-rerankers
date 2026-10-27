"""Local evidence viewer for the relevance-scoring lab.

Ask a question, compare baseline retrieval with LLM reranking, and inspect
source chunks and citation labels. Install requirements, then run: python app.py
"""

import os
import re
import threading
import time

import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

TEST_QUERIES = [
    "What AI practices are prohibited under the EU AI Act?",
    "What are the seven requirements for trustworthy AI?",
    "What is the difference between human-in-the-loop and human-on-the-loop?",
    "What fines apply for using a prohibited AI practice?",
    "Is social scoring of citizens allowed?",
]

index_name = "lab-rerankers"
vectorstore = None
llm = None
rag_chain = None
LLM_CALLS = 0
RUN_LOCK = threading.Lock()

rag_prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context.
After each claim, cite the source label in square brackets exactly as it appears in the context.
If the context does not contain the answer, reply exactly: "I don't know based on the sources."

Context:
{context}

Question: {question}

Answer:"""
)


def missing_api_keys():
    """Return the required environment variable names that are missing."""
    return [name for name in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not os.getenv(name)]


def require_api_keys():
    """Stop with a useful message when a required key is missing."""
    missing = missing_api_keys()
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"Missing {names}. Add the required key(s) to .env and run again.")


def initialize():
    """Connect to the existing index and prepare the answer model."""
    global vectorstore, llm, rag_chain
    require_api_keys()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name="lab-rerankers", embedding=embeddings)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rag_chain = rag_prompt | llm | StrOutputParser()


def baseline_search(query, k=5, metadata_filter=None):
    """Plain vector search: return document and cosine similarity pairs."""
    return vectorstore.similarity_search_with_score(query, k=k, filter=metadata_filter)


def llm_relevance_score(query, doc):
    """Ask the LLM to score one chunk's relevance from 0 to 1."""
    global LLM_CALLS
    prompt = (
        f"Rate the relevance of this document to the query '{query}' on a scale of 0-1.\n"
        "Reply with only the number, for example 0.7.\n\n"
        f"Document:\n{doc.page_content}"
    )
    response = llm.invoke(prompt)
    LLM_CALLS += 1
    match = re.search(r"\d+(?:\.\d+)?", response.content)
    try:
        score = float(match.group()) if match else 0.5
    except ValueError:
        score = 0.5
    return min(max(score, 0.0), 1.0)


def llm_rerank(query, k_initial=10, k_final=5, w_sim=0.3, w_llm=0.7,
               verbose=False, metadata_filter=None):
    """Retrieve candidates, score them, and return top chunks with every score."""
    candidates = baseline_search(query, k=k_initial, metadata_filter=metadata_filter)
    scored = []
    for doc, sim in candidates:
        llm_score = llm_relevance_score(query, doc)
        combined = w_sim * sim + w_llm * llm_score
        scored.append((doc, combined, sim, llm_score))
        if verbose:
            print(f"  {doc.metadata['chunk_id']:<16} sim={sim:.3f}  llm={llm_score:.2f}  combined={combined:.3f}")
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored[:k_final]


def source_label(doc):
    """Return the same human-readable citation label used in the notebook."""
    metadata = doc.metadata
    if "page" in metadata:
        label = f"{metadata['source_name']}, p.{int(float(metadata['page']))}"
        if "article" in metadata:
            label += f", {metadata['article']}"
        elif "annex" in metadata:
            label += f", {metadata['annex']}"
        return label
    return f"{metadata['source_name']}, {metadata['chunk_id']}"


def format_context(results):
    """Join chunks into a context string, each with its citation label."""
    return "\n\n".join(
        f"[{source_label(row[0])}]\n{row[0].page_content}" for row in results
    )


def answer(query, retrieve_fn=baseline_search):
    """Retrieve, format, and answer. Return answer text and retrieved rows."""
    global LLM_CALLS
    if rag_chain is None:
        raise RuntimeError("Call initialize() before asking a question.")
    results = retrieve_fn(query)
    text = rag_chain.invoke({"context": format_context(results), "question": query})
    LLM_CALLS += 1
    return text, results


def build_filter(source_choice, article_text):
    """Turn the dropdown and optional article into Pinecone filter syntax."""
    conditions = []
    if source_choice == "Regulation only":
        conditions.append({"source_type": {"$eq": "regulation"}})
    elif source_choice == "Podcast only":
        conditions.append({"source_type": {"$eq": "podcast"}})
    if article_text and article_text.strip():
        conditions.append({"article": {"$eq": article_text.strip()}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def run_question(question, method, source_choice, article_text):
    """Run one question and prepare every visible evidence output."""
    if not question or not question.strip():
        return "Enter a question first.", "", [], gr.Dropdown(choices=[], value=None, label="Pick a chunk to inspect"), "", "0.00 seconds | 0 LLM calls", {}

    metadata_filter = build_filter(source_choice, article_text)
    with RUN_LOCK:
        calls_before = LLM_CALLS
        started = time.perf_counter()
        if method == "Baseline":
            retrieve_fn = lambda query: baseline_search(query, metadata_filter=metadata_filter)
        else:
            retrieve_fn = lambda query: llm_rerank(query, metadata_filter=metadata_filter)
        answer_text, results = answer(question.strip(), retrieve_fn=retrieve_fn)
        seconds = time.perf_counter() - started
        call_count = LLM_CALLS - calls_before

    evidence_rows = []
    chunk_choices = []
    chunk_texts = {}
    citation_lines = []
    for rank, result in enumerate(results, 1):
        doc = result[0]
        metadata = doc.metadata
        if method == "Baseline":
            similarity = float(result[1])
            llm_score = ""
            combined_score = ""
        else:
            combined_score = float(result[1])
            similarity = float(result[2])
            llm_score = float(result[3])
        page_or_id = f"p.{int(float(metadata['page']))}" if "page" in metadata else metadata.get("chunk_id", "")
        snippet = " ".join(doc.page_content.split())[:150]
        chunk_id = metadata.get("chunk_id", "")
        evidence_rows.append([
            rank, chunk_id, metadata.get("source_name", metadata.get("source_type", "")),
            page_or_id, metadata.get("section", ""), metadata.get("article", ""),
            similarity, llm_score, combined_score, snippet,
        ])
        chunk_choices.append((f"{rank}: {chunk_id}", chunk_id))
        chunk_texts[chunk_id] = doc.page_content
        citation_lines.append(f"- [{source_label(doc)}] maps to `{chunk_id}`")

    selected = chunk_choices[0][1] if chunk_choices else None
    full_text = chunk_texts.get(selected, "No chunks were retrieved.")
    citation_trace = "\n".join(citation_lines) if citation_lines else "No evidence chunks were retrieved."
    timing = f"{seconds:.2f} seconds | {call_count} LLM calls"
    return answer_text, citation_trace, evidence_rows, gr.Dropdown(choices=chunk_choices, value=selected, label="Pick a chunk to inspect"), full_text, timing, chunk_texts


def make_example_handler(example_query):
    """Create a button handler that runs one fixed test query."""
    def handle_example(method, source_choice, article_text):
        outputs = run_question(example_query, method, source_choice, article_text)
        return (example_query, *outputs)
    return handle_example


def show_selected_chunk(chunk_id, chunk_texts):
    """Show the selected chunk's complete text."""
    if not chunk_id:
        return "Pick a chunk to inspect its full text."
    return chunk_texts.get(chunk_id, "The selected chunk is not in the current results.")


def build_demo():
    """Build the local Gradio interface."""
    with gr.Blocks(title="Relevance scoring evidence viewer") as demo:
        gr.Markdown("# Relevance scoring evidence viewer\nAsk a question and inspect the chunks behind the answer.")
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Ask about the EU AI Act or the podcast", scale=3)
            method = gr.Radio(["Baseline", "LLM reranked"], value="Baseline", label="Method")
        with gr.Row():
            source_filter = gr.Dropdown(
                ["All", "Regulation only", "Podcast only"], value="All", label="Source filter"
            )
            article_filter = gr.Textbox(label="Article filter (optional)", placeholder="Article 99")
            ask_button = gr.Button("Ask", variant="primary")

        answer_output = gr.Markdown(label="Answer")
        trace_output = gr.Markdown(label="Citation trace")
        evidence_output = gr.Dataframe(
            headers=["Rank", "Chunk ID", "Source", "Page or chunk_id", "Section", "Article",
                     "Similarity", "LLM score", "Combined score", "First 150 characters"],
            interactive=False,
            wrap=True,
            label="Retrieved evidence",
        )
        with gr.Row():
            selected_chunk = gr.Dropdown(choices=[], label="Pick a chunk to inspect")
            full_text = gr.Textbox(label="Full chunk text", lines=12, interactive=False)
        timing_output = gr.Markdown()
        chunk_texts = gr.State({})

        ask_button.click(
            fn=run_question,
            inputs=[question, method, source_filter, article_filter],
            outputs=[answer_output, trace_output, evidence_output, selected_chunk, full_text, timing_output, chunk_texts],
        )
        selected_chunk.change(
            fn=show_selected_chunk,
            inputs=[selected_chunk, chunk_texts],
            outputs=full_text,
        )
        gr.Markdown("## Example questions")
        for query in TEST_QUERIES:
            button = gr.Button(f"Example: {query}")
            button.click(
                fn=make_example_handler(query),
                inputs=[method, source_filter, article_filter],
                outputs=[question, answer_output, trace_output, evidence_output, selected_chunk,
                         full_text, timing_output, chunk_texts],
            )
    return demo


CSS = """
.gradio-container { max-width: 1200px !important; }
"""


def main():
    """Check keys, connect to Pinecone, and launch on this computer."""
    require_api_keys()
    initialize()
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", theme=gr.themes.Soft(), css=CSS, share=False)


if __name__ == "__main__":
    main()
