import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = r"D:\PDF_QA\models\models--deepset--roberta-base-squad2\snapshots\adc3b06f79f797d1c575d5479d6f5efe54a9e3b4"

question_answer = pipeline("question-answering", model="deepset/roberta-base-squad2")

# question_answer = pipeline("question-answering", model=model_path)

def read_file_context(file_obj):
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as file:
            context = file.read()
            return context
    except Exception as e:
        return f"An error occurred: {e}"

def get_answer(file, question):
    context = read_file_context(file)
    answer = question_answer(question=question, context=context)
    return answer["answer"]

demo = gr.Interface(fn = get_answer, inputs=[gr.File(label="Upload Your File"), gr.Textbox(label="Input Your Question", lines=11)], outputs=[gr.Textbox(label="Answer Text", lines=3)], title="PDF Question and Answer", description="This application will be used to answer the question from the provided PDF")
demo.launch()