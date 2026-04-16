#import "@preview/ctheorems:0.1.0": *

#let base_templ(doc) = [
  #set text(font: "New Computer Modern", lang: "ru")
  #show raw: set text(font: "New Computer Modern Mono")
  #set par(justify: true)
  #show heading: set block(above: 1.4em, below: 1em)
  
  // #show math.ast: math.dot.op
  
  // #set heading(numbering: "1.a.1")
  #show heading: it => {
    it
    // linebreak()
  }

  #doc
]

#let theorem = thmbox(
  "theorem", "Теорема", fill: rgb("#E1F5FE"), stroke:rgb("#4FC3F7")
)

#let task = thmbox(
  "task", "Задача", fill: rgb("#ccff99"), stroke:rgb("#99ff66")
).with(numbering: "1.A")

#let proof = thmplain(
  "proof",
  "Доказательство",
  base: "theorem",
  bodyfmt: body => [#body #h(1fr) $qed$],
).with(numbering: none)

#let solution = thmplain(
  "solution",
  "Решение",
  base: "task"
).with(numbering: none)

#let lim = $limits(lim)$

#let limoo = $lim_(n -> oo)$

#let eps = $epsilon$
