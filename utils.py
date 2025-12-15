import re


def chunk_rst_text(filepath, filename):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by header pattern
    sections = re.split(r'\n(?=[^\n]+\n[=\-~^"]{3,}\n)', content)

    chunks = []
    for i, section in enumerate(sections):
        if section.strip():
            # Extract header (first line before the marker)
            lines = section.split('\n')
            header = lines[0].strip() if lines else "Untitled"

            chunks.append({
                'filename': filename,
                'source': str(filepath),
                'id': f"{filename}_chunk_{i}",
                'header': header,
                'text': section.strip(),
            })

    return chunks

def print_chunk(chunk):
    line_length = 95
    print(f"{'=' * line_length}")
    print(f"[ID] {chunk['id']}")
    print(f"{'=' * line_length}")
    for key in chunk.keys():
        if key != 'text':
            print(f"{key}: {chunk[key]}")
    print(f"\nTEXT:\n{'-' * line_length}")
    print(f"{chunk['text']}")
    print(f"{'-' * line_length}")
    print("\n\n")



