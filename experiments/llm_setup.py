from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_debug, set_verbose
import getpass
import os

# Global LangChain settings
set_debug(False)
set_verbose(False)

GPT_MODEL = "gpt-5-mini"
GEMINI_MODEL = "gemini-2.5-flash"

def get_llms(provider="openai"):
    """
    Returns a tuple of (lab_tech_llm, triage_llm, diagnose_llm, critique_llm, synthesizer_llm)
    based on the specified provider ('openai' or 'gemini').
    """
    if provider.lower() == "openai":
        lab_tech_llm = ChatOpenAI(
            model=GPT_MODEL,
            temperature=0.0,
            use_responses_api=True
        )

        triage_llm = ChatOpenAI(
            model=GPT_MODEL,
            reasoning={"effort": "low"},
            max_completion_tokens=896,
            temperature=0.2,
            use_responses_api=True
        )

        diagnose_llm = ChatOpenAI(
            model=GPT_MODEL,
            reasoning={"effort": "low"},
            temperature=0.3,
            max_completion_tokens=1600,
            use_responses_api=True
        )

        critique_llm = ChatOpenAI(
            model=GPT_MODEL,
            reasoning={"effort": "low"},
            temperature=0.3,
            max_completion_tokens=1300,
            use_responses_api=True
        )

        synthesizer_llm = ChatOpenAI(
            model=GPT_MODEL,
            reasoning={"effort": "low"},
            temperature=0.3,
            max_completion_tokens=1300,
            use_responses_api=True
        )
        return lab_tech_llm, triage_llm, diagnose_llm, critique_llm, synthesizer_llm

    elif provider.lower() == "gemini":

        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

        lab_tech_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.0,
            top_p=0.95
        )

        triage_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            max_output_tokens=1800,
            thinking_budget=1000,
            temperature=0.2,
            top_p=0.95
        )

        diagnose_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            max_output_tokens=2400,
            thinking_budget=1000,
            temperature=0.3,
            top_p=0.95
        )

        critique_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            max_output_tokens=1800,
            thinking_budget=1000,
            temperature=0.3,
            top_p=0.95
        )

        synthesizer_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            max_output_tokens=1800,
            thinking_budget=1000,
            temperature=0.3,
            top_p=0.95
        )
        return lab_tech_llm, triage_llm, diagnose_llm, critique_llm, synthesizer_llm
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'gemini'.")
