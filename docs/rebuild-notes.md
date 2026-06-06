# Rebuild notes

An honest log of where this project came from, what I changed coming back to it, and
what actually happened. Written plainly so I can turn it into a post later.

## the original

I started this in undergrad with a couple of friends. The idea was genuinely fun: the
immune system is brilliant at noticing local threats, so could an immune-inspired
algorithm, the Dendritic Cell Algorithm, notice local *unfairness* in a model the same
way? We read around it, surveyed the fairness tools (Fairlearn, AIF360, Aequitas, the
What-If Tool), picked a dataset, and started benchmarking DCA against some classifiers.

Then the team thinned out and it became mine, and I hit the wall every undergrad hits: I
had a compelling metaphor and a half-built comparison, but no way to actually *prove*
whether the immune idea added anything. I even wrote in my notes at the time that it
might be "just relabeling bias as danger signals." I suspected the weakness. I didn't yet
have the tools to resolve it, so it stayed a nice idea that never quite landed.

## coming back to it

What changed is not the idea, it is what I now reach for first. Three years ago the idea
was the project. This time the *test* was the project.

- **I asked a sharper question.** Not "is bias like danger," which is unfalsifiable, but
  "does this catch local pockets of bias that global metrics and subgroup auditors miss?"
  That is something you can actually measure.
- **I built the experiment I didn't know how to build then.** Inject a known bias into a
  known region, then measure how well each detector recovers it. Ground truth. Without it,
  every claim about a bias detector is just vibes.
- **I gave the immune idea its real shot.** DCA's actual strength is fusing several weak
  signals, so I built four local fairness signals plus a counterfactual PAMP, and put the
  neighbourhoods in a bias-relevant subspace so a localized pocket would not be diluted.
- **I compared against a real competitor**, Slice Finder, not just classifiers.

## what broke, and the honest finding

The first naive version of DCA flagged nothing: on a static table, random sampling makes
every cell average out to "normal." That pointed straight at the fix, local-neighbourhood
sampling, which is also the whole local-pocket idea. Good sign.

But the deeper finding held up under everything I threw at it: **the immune machinery does
not beat a simple average of the same signals.** Across four datasets, two pocket shapes,
and two noise regimes, the plain fused signal matched or beat DCA every time, even in the
noisy regime I built specifically for aggregation to help. The reason is clean: DCA's
maturation step collapses a graded signal into a binary context and pools it, which throws
away resolution that the average keeps.

The flip side is the real win: that simple fused local-fairness signal is competitive with
Slice Finder, a purpose-built auditor, and it is far simpler and interpretable.

## what I gained

This is the opposite of wasted time. I went back to a loose undergrad idea and turned it
into a rigorous study with a clear, defensible result and a method that actually works. I
also resolved the doubt my younger self flagged but couldn't settle, with evidence instead
of a hunch. The lesson I am keeping: test before you believe, and a negative result you
understand is worth more than a positive one you don't.
