from annotated_text import annotation
from json import JSONDecodeError

from markdown import markdown
import streamlit as st

from utils.haystack import start_haystack
from utils.ui import reset_results, set_initial_state, sidebar
from utils.config import TWITTER_BEARER
from haystack.utils import print_answers

set_initial_state()

pipe = start_haystack()

st.write("# Any question about Game of Thrones?")

search_bar, button = st.columns(2)
    # Search bar
with search_bar: 
    question = st.text_input("Please enter your question", on_change=reset_results)

with button: 
    st.write("")
    st.write("")
    run_pressed = st.button("Ask question")
        
if run_pressed:
    prediction = pipe.run(
        query=question,
        params={
            "Retriever": {"top_k": 10},
            "Reader": {"top_k": 5}
        }
    )
    allAnswersContext = ""
    for answer in prediction["answers"]:
        st.caption("answer - score:" + str(answer.score) +"")
        st.write( answer.answer)
        
        allAnswersContext += answer.answer + "  " + answer.context
        

    from utils.summarizer import summarize 
    
    st.write("## Summary")
    with st.spinner("🔎"):
        summary = summarize(question, allAnswersContext)
        st.write("-------")
        st.write(summary[0])
        st.write("-------")
        
    
