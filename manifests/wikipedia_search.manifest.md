# Tool: wikipedia_search

## Intent
Search Wikipedia and return a summary of the article most relevant to the query.
Use this tool when you need factual background on a topic: people, events, concepts, papers, or organizations.

## Usage Patterns
- Pass concise 2-4 word queries using proper nouns: `"Geoffrey Hinton"`, `"backpropagation"`, `"GPT-3"`
- For landmark papers, include the common name or author + topic: `"Attention Is All You Need"`, `"Turing test"`
- For organizations or labs: `"DeepMind"`, `"OpenAI"`, `"Bell Labs AI"`
- Chain calls to build context — search for a person, then a concept they pioneered

## Known Failure Modes
- Verbose full-sentence queries often return wrong or unrelated articles
- Disambiguation: generic terms like `"neural network"` may hit a disambiguation page — add a qualifier like `"artificial neural network"`
- Very recent events may not have Wikipedia articles yet

## Output Interpretation
- The summary is the lead section of the Wikipedia article (typically 3-10 sentences)
- A `DisambiguationError` means your query was too generic — retry with a more specific term
- A `PageError` means no article matched — rephrase or try an alternate name
