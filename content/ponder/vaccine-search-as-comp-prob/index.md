---
title: "Vaccine Search as a Computational Problem"
date: 2021-02-06
tags: ["Machine Learning", "Covid-19", "Computational Biology"]
author: "Franz Louis Cesista"
description: "A thought dump on mRNA vaccines and the future of computational biology"
summary: "A thought dump on mRNA vaccines and the future of computational biology"
cover:
    image: "cover.jpg"
    alt: "Cover"
editPost:
    URL: "https://ponder.substack.com/p/vaccine-search-as-comp-prob"
    Text: "Crossposted on Ponder"
---

## Traditional vs. BioNTech’s mRNA vaccines

Despite what you might think, the human body is excellent at preventing diseases. When a virus enters our bloodstream, our immune system produces antibodies to eliminate it. The problem is, for viruses our immune system hasn't dealt with before, this process is often slow, and we could die before it takes full effect.

We used to speed up this process with traditional vaccines by injecting ourselves with dead or weaker versions of the virus. Think of it as "warning" our immune system so it could stockpile war machines for when the "real" virus comes along.

BioNTech’s Covid-19 vaccine is brilliant and simple. We don't need to inject the whole virus - just a "message" to our cells to produce the virus's "spike protein." We encode this "message" in an mRNA. And the spike protein is what makes our immune system go full panic mode like: "OMG!! I recognize this protein! Let’s build the antibodies real quick!!"

What fascinates me with the new vaccine is how much innovative trickery was involved in developing it[1]. Here's a non-exclusive list:

1. The researchers modified the mRNA's U-molecule a bit to effectively "cloak" it from our immune system. This way, our "message" could safely reach our cells for them to produce the spike proteins.

2. They stuffed the mRNA with [C](https://en.wikipedia.org/wiki/Cytosine) and [G](https://en.wikipedia.org/wiki/Guanine) molecules to fool our cells to prioritize building the spike proteins. They did this in such a way that the local structure of the spike protein wouldn't change.

3. They added a double [Proline fragment](https://en.wikipedia.org/wiki/Proline#Properties_in_protein_structure) as some form of splint to stabilize the protein spike. They needed to do this because the original protein spike easily collapses when disconnected from the virus. And

4. They then used [DNA printers](https://codexdna.com/products/bioxp-system/) to mass-produce the vaccines. They also open-sourced the sequence [here](https://mednet-communities.net/inn/db/media/docs/11889.doc).

---

## The search for vaccines as a computational problem

Current experimental biology research, like most sciences, is like a [guided search algorithm](https://en.wikipedia.org/wiki/Guided_Local_Search): we know where we want to go, but we don't know how to get there. So we try out the next best thing and the next best thing after it, and so on.

Why don't we automate this process?

I mean, let's do what the engineers do. Let's not put theories on a pedestal and simulate the stuff until we find out what works best. We could:

- Generate thousands of U-molecule modifications and pick the one with the best "cloak."

- Create millions of mRNA alterations, fold them using [AlphaFold](https://deepmind.com/blog/article/alphafold-a-solution-to-a-50-year-old-grand-challenge-in-biology), and find which one is the strongest but still close enough to the original protein. And even

- Make molecular machines just for fun!

---

The age of computational biology has begun.

In a few years, I expect us to print custom-made DNA and mRNA like how we 3D print stuff today. Think about all the possibilities! Home-made insulin, personalized drugs, plant & pet enhancers, and so on. The next generation will only need to go to drugstores to buy biological ink and print what they need at home.

---

## Possible dangers ahead

I believe democratizing these technologies is necessary, inevitable, and would have a net-positive effect. But I’m concerned this would also make it easier for terrorists to make bioweapons. And doing so is far simpler and cheaper than finding and manufacturing its cure[2].

With this tech, a terrorist organization could develop a bioweapon in some lawless part of the world, open-source the whole sequence, and let their sleeper agents around the world reproduce them. With the necessary hardware and internet connection, a lone wolf could infect and kill hundreds of millions. Simultaneously, the organization does psych ops on social media networks to divide and confuse us.

If we're not careful, bioterrorism could be worse than nuclear war[3].

---

What do you think?

<center><iframe src="https://ponder.substack.com/embed" width="480" height="320" style="border:1px solid #EEE; background:white;" frameborder="0" scrolling="no"></iframe></center>

---

[1] Hubert, Bert. [Reverse Engineering the Source Code of the BioNTech-Pfizer SARS-CoV-2 Vaccine](https://berthub.eu/articles/posts/reverse-engineering-source-code-of-the-biontech-pfizer-vaccine/).

[2] Neubert, Jonas. [Exploring the Supply Chain of the Pfizer/BioNTech and Moderna COVID-19 vaccines](https://blog.jonasneubert.com/2021/01/10/exploring-the-supply-chain-of-the-pfizer-biontech-and-moderna-covid-19-vaccines/).

[3] Taleb, Nassim. [Nassim Nicholas Taleb Sees Greater Risks Than Nuclear War](https://www.youtube.com/watch?v=7V7W46sOt38&feature=youtu.be). Interview with Schatzker, Erik on YouTube, uploaded by Bloomberg Markets and Finance.
