# build_contextual_TEST.py — γρήγορο τεστ με 1 μικρό βιβλίο
from dotenv import load_dotenv
from rag_engine import add_contextual_documents
load_dotenv()

# Μόνο 1 βιβλίο για τεστ (το πιο μικρό)
add_contextual_documents(
    "data",
    only_files=["Six Easy Pieces_ Essentials of Physics Explained by Its Most Brilliant Teacher by Richard P. Feynman.pdf"],
    collection_suffix="test"   # ξεχωριστή test collection
)