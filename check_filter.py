from rag_engine import load_index, get_chat_engine, list_indexed_files

index = load_index()
files = list_indexed_files()
print("Available:", files)

# Φτιάξε engine με filter σε ΕΝΑ αρχείο
engine = get_chat_engine(index, file_names=[files[0]])
response = engine.chat("test question")

# Έλεγξε: ΟΛΕΣ οι πηγές πρέπει να είναι από files[0]
for node in response.source_nodes:
    print(node.metadata.get("file_name"))   # πρέπει όλα να είναι files[0]