import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
from huggingface_hub import login
import numpy as np
import random

RANDOM_SEED = 42
sys.path.insert(
    0,
    os.path.join(
       os.path.dirname(os.path.dirname(__file__))
    )
)
# print(sys.path)
from raptor import RetrievalAugmentation
# from raptor.EmbeddingModels import SBertEmbeddingModel
from raptor import (
    BaseSummarizationModel,
    BaseQAModel,
    BaseEmbeddingModel,
    RetrievalAugmentationConfig
)
sys.path.pop(0)


def get_parser():
    parser = argparse.ArgumentParser(description="demo raptor")
    parser.add_argument(
        "--tree_path",
        default="cinderella_tree.pkl",
        help="where to save the tree"
    )
    return parser


# You can define your own Summarization model by extending the base Summarization Class.
class GEMMASummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="google/gemma-2b-it"):
        # Initialize the tokenizer and the pipeline for the GEMMA model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarization_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # Use "cpu" if CUDA is not available
        )

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        messages=[
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate the summary using the pipeline
        apply_random_seed(RANDOM_SEED)
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            # do_sample=True,
            do_sample=False, # tmp
            temperature=0.7,
            # temperature=0.0, # tmp
            top_k=50,
            top_p=0.95
        )

        # Extracting and returning the generated summary
        summary = outputs[0]["generated_text"].strip()
        # remove technical prefix
        split = summary.split("start_of_turn>model\n")
        summarization_prompt = "\n\n".join(split[:-1])
        summary = split[-1].strip()
        return summary, summarization_prompt


class GEMMAQAModel(BaseQAModel):
    def __init__(self, model_name= "google/gemma-2b-it"):
        # Initialize the tokenizer and the pipeline for the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        )

    def answer_question(self, context, question):
        # Apply the chat template for the context and question
        messages=[
              {"role": "user", "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"}
        ]
        print("Context: ", context)
        print("Question: ", question)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Generate the answer using the pipeline
        outputs = self.qa_pipeline(
            prompt,
            max_new_tokens=256,
            # do_sample=True,
            do_sample=False, # tmp
            temperature=0.7,
            # temperature=0.0, # tmp
            top_k=50,
            top_p=0.95
        )

        # Extracting and returning the generated answer
        answer = outputs[0]["generated_text"][len(prompt):]
        return answer


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


def apply_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to suppress warning
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():

    args = get_parser().parse_args()

    apply_random_seed(RANDOM_SEED)

    # NOTE: An OpenAI API key must be set here for application initialization, even if not in use.
    # If you're not utilizing OpenAI models, assign a placeholder string (e.g., "not_used").
    os.environ["OPENAI_API_KEY"] = "your-openai-key"


    # Cinderella story defined in sample.txt
    sample_text_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'sample.txt')
    with open(sample_text_path, 'r') as file:
        text = file.read()

    print(text[:100])

    # if you want to use the Gemma, you will need to authenticate with HuggingFace, Skip this step, if you have the model already downloaded

    with open('/home/oh/arubinstein17/.config/hugging_face/hf.yaml', 'r') as file:
        token = file.read()
        token = token.split(":")[1].strip()
    # print(token)
    login(token=token)

    tree_path = args.tree_path
    RAC = RetrievalAugmentationConfig(
        summarization_model=GEMMASummarizationModel(),
        qa_model=GEMMAQAModel(),
        embedding_model=SBertEmbeddingModel(),
        tr_threshold=0.9,
        tb_max_tokens=100,
        tb_summarization_length=200
    )
    RA = RetrievalAugmentation(config=RAC, tree=tree_path)

    # RA = RetrievalAugmentation(
    #     embedding_models={"SBert": SBertEmbeddingModel()},
    #     cluster_embedding_model="SBert"
    # )
    # # {"OpenAI": OpenAIEmbeddingModel()}

    if not os.path.exists(tree_path):
        # construct the tree
        RA.add_documents(text, use_multithreading=False)

        RA.save(tree_path)

    # question = "How did Cinderella reach her happy ending?"
    # question = "What was a special role of the shoe in the story?"
    # question = "Which item was used to identify Cinderella?"
    question = "What was the cause of Evelyn's symptoms?"

    answer = RA.answer_question(
        question=question,
        top_k=1,
        collapse_tree=False,
        start_layer=1, # goes from top to bottom (descending); from this layer the seeing set of nodes is taken
    )

    print("Answer: ", answer)


if __name__ == "__main__":
    main()
