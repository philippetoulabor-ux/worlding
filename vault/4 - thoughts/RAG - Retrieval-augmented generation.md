


2026-06-23 17:17

tags: [[Agents and LLMs]], [[OS]]


https://en.wikipedia.org/wiki/Retrieval-augmented_generation

Retrieval-augmented generation (RAG) ist ein Verfahren, bei dem große Sprachmodelle (LLMs) vor der Antwort **gezielt externe Informationen abrufen**, statt sich nur auf ihr statisches Trainingswissen zu verlassen. Dazu werden relevante Dokumente aus Datenbanken, internen Firmenquellen oder dem Web gesucht und zusammen mit der Nutzerfrage als erweiterter Prompt ins Modell gegeben. So können Chatbots aktuelles oder domänenspezifisches Wissen nutzen, ohne das Modell ständig neu zu trainieren.

Technisch werden die Referenzdaten meist in Vektoren (Embeddings) umgewandelt und in einem Vektor­datenbank‑Index gespeichert. Bei einer Anfrage wählt ein Retriever die passendsten Textstücke aus, die dann via Prompt Engineering in die Modell­eingabe „eingestopft“ werden (Prompt Stuffing), damit das Modell diese frischen Infos gegenüber seinem alten Trainingsstand priorisiert.

RAG soll Halluzinationen reduzieren und Fakten­treue erhöhen, löst das Problem aber nicht vollständig: Das Modell kann kontextlose oder irreführende Quellen falsch interpretieren, widersprüchliche Informationen vermischen oder trotz fehlender Grundlage weiterhin „erfinden“. Zudem bleibt ein gewisser Bedarf an Modell­weiterentwicklung und ‑training bestehen, etwa um Unsicherheit explizit zu erkennen.

Der Artikel beschreibt mehrere Verbesserungs­hebels entlang der Pipeline: bessere Encoder (dichte/sparse Vektoren, Late Interaction, Hybrid-Ansätze), retriever-zentrierte Methoden (Vortraining, supervised Optimierung, Reranking), architektonische Anpassungen der Sprachmodelle (z.B. Retro/Retro++), Chunking-Strategien für unterschiedliche Dateitypen sowie Hybrid Search (Kombination aus Vektor- und klassischer Volltextsuche). Ein eigenes Unterkapitel widmet sich „RAG Poisoning“: selbst korrekte, aber irreführend eingebettete oder kontextlose Inhalte können zu falschen Schlussfolgerungen führen, wenn das Modell sie nicht richtig einordnet.

Kurz: RAG ist heute ein zentrales Muster, um LLMs faktennäher, aktueller und anwendungs­spezifischer zu machen, bleibt aber anfällig für Fehlinterpretationen, schlechte Retrievalqualität und manipulierte oder missverstandene Quellen.

gutes youtube video!
RAG Explained For Beginners