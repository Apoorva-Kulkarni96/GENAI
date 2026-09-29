


def build_context(results):
    return "\n\n".join(
        result["document"]["text"]
        for result in results
    )

    




