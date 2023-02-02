# pTreeTSL annotation notes

## Naming conventions

Empty categories are given a mnemonic name, in the style of the Penn Historical Corpora.
- Ex. embedded interrogative C is "C-emb-wh"

## Phrase structure

Standard Minimalist analysis is used by default.
- Noun phrases are DPs
	- Names and plurals have an empty D head
	- who/what are D heads
- Verb phrases are headed by little v (VP shell analysis)
- Finite clauses are CPs

## Lexical selection

Words appearing with multiple selection frames are given distinct labels.
- Ex. "book" (-N) vs "book-pobj" (+P -N), used for "book about"
- This only affects nouns at present.
- For consistency, nouns and verbs with PP objects are always labeled as such.

## A'-movement

Wh-movement and RC-movement are treated uniformly, with the wh-operator moving to SpecCP
- Exception: "whether" is treated as a head

## Adjuncts

Adjuncts are treated as dependents of heads.
- Post-verbal PPs are inside VP, as in a VP-shell analysis
- Preverbal adverbs are adjuncts to vP
- where/when are adjuncts to vP
- how/why are adjuncts to TP
- All PP dependents of N are treated as arguments.

In general, adjuncts are placed according to their surface order w.r.t. other dependents.
- Exception: wh-phrase adjuncts are placed in initial position

## Control

Subject of control predicate is PRO.
No further annotation.

## Simplifications

Compounds and some modified nouns are treated as a single lexical item.
Words are joined with an underscore.
- Ex. compound: police officer
- Ex. modified noun: famous actor