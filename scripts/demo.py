import os
import sys
import argparse


import sys
import os

sys.path.insert(
    0,
    os.path.join(
       os.path.dirname(os.path.dirname(__file__))
    )
)
# print(sys.path)
from raptor import RetrievalAugmentation
from raptor.EmbeddingModels import SBertEmbeddingModel
sys.path.pop(0)


def get_parser():
    parser = argparse.ArgumentParser(description="demo raptor")
    parser.add_argument(
        "--result_path",
        default="tmp_result.out",
        help="where to save the results"
    )
    return parser


def main():

    args = get_parser().parse_args()

    # NOTE: An OpenAI API key must be set here for application initialization, even if not in use.
    # If you're not utilizing OpenAI models, assign a placeholder string (e.g., "not_used").
    os.environ["OPENAI_API_KEY"] = "your-openai-key"


    # Cinderella story defined in sample.txt
    sample_text_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'sample.txt')
    with open(sample_text_path, 'r') as file:
        text = file.read()

    print(text[:100])

    RA = RetrievalAugmentation(
        embedding_models={"SBert": SBertEmbeddingModel()},
        cluster_embedding_model="SBert"
    )
    # {"OpenAI": OpenAIEmbeddingModel()}

    # construct the tree
    RA.add_documents(text)


if __name__ == "__main__":
    main()
