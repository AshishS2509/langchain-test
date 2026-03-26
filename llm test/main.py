from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

load_dotenv()


def main():
    information = """
    Chanakya (flourished 300 bce) was a Hindu statesman and philosopher who wrote a classic treatise on polity, Artha-shastra (“The Science of Material Gain”), a compilation of almost everything that had been written in India up to his time regarding artha (property, economics, or material success).

    He was born into a Brahman family and received his education at Taxila (now in Pakistan). He is known to have had a knowledge of medicine and astrology, and it is believed he was familiar with elements of Greek and Persian learning introduced into India by Zoroastrians. Some authorities believe he was a Zoroastrian or at least was strongly influenced by that religion.

    Chanakya became a counselor and adviser to Chandragupta (reigned c. 321–c. 297), founder of the Mauryan empire of northern India, but lived by himself. He was instrumental in helping Chandragupta overthrow the powerful Nanda dynasty at Pataliputra, in the Magadha region.
    """

    summary_prompt = """
    given this information: {information} about the person, I want you to create:
    A short summary and two interesting facts about the person.
"""
    summary_prompt_template = PromptTemplate(
        input_variables={"information"}, template=summary_prompt
    )
    llm = OllamaLLM(model="gemma3:270m")
    chain = summary_prompt_template | llm
    resp = chain.invoke(input={"information": information})
    print(resp)


if __name__ == "__main__":
    main()
