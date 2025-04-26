import os
import re
import sys

from matplotlib import pyplot as plt
from langchain.schema import Document

def clean_business_conduct_policy(docs, n_remove_first_lines = -1, n_discard_pages=[]):
    
    cleaned_docs = []

    for page_idx, doc in enumerate(docs):        
        if page_idx+1 in n_discard_pages:
            continue

        
        if n_remove_first_lines != -1:
            content = doc.page_content
            lines = content.split('\n')

            cleaned_content = '\n'.join(lines[n_remove_first_lines+1:])


            cleaned_doc = Document(
                metadata=doc.metadata,
                page_content=cleaned_content
            )

            cleaned_docs.append(cleaned_doc)
        
        else:

            cleaned_docs.append(doc)
    
    return cleaned_docs 


def clean_gemma_response(decoded_output, user_message):
    if user_message in decoded_output:
        return decoded_output.split(user_message, 1)[-1].strip()
    else:
        return decoded_output.strip()