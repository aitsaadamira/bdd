# Brexit Gold Standard

This repository contains the Brexit Twitter Gold Standard produced by the
[SSIX Project](http://ssix-project.eu/).

[![SSIX](https://bytebucket.org/ssix-project/brexit-gold-standard/raw/master/assets/ssix-navbar.png)](http://ssix-project.eu/)

---

[TOC]

---


## Preparation

You will need to prepare a few basic things:

1. Clone this repository `git clone https://bitbucket.org/ssix-project/brexit-gold-standard.git ssix-brexit-gold-standard`
2. Move into the cloned repository: `cd ssix-brexit-gold-standard/`
3. You need to have both a `python` interpreter and `pip` installed
4. Install required dependecies: `pip install -r requirements.txt --upgrade`
5. [Create a new Twitter App](https://apps.twitter.com/app/new), then create a
   read-only Access Token, and fill in the missing details at `brexit-gs.yml`


## Rebuild full sample

The [published sample](https://bitbucket.org/ssix-project/brexit-gold-standard/src/master/brexit-sample-20160506-annotated.json)
is anonymized to not distribute any personal detail.

To rebuild the full dataset, you need to execute the following script:

    python rebuild.py brexit-sample-20160506-annotated.json

This process could take a while, depending on the pause configured at `brexit-gs.yml`
to obey the [Twitter API rate limits](https://dev.twitter.com/rest/public/rate-limiting).

At the end it willl create a `brexit-sample-20160506-annotated-full.json` with the full dataset rebuilt,
including both the original tweet data and annotations.


## Annotations

Each Tweet is annotated with the following information:

1. `sentiment`: one of "stay", "leave", "undecided", "no sentiment/don't care", "irrelevant", or "" (left blank).
2. `strength`: for “stay” or “leave”: an integer between `1` (very weak) and `5` (very strong) expressing the strength of the opinion. For all other tweets, `0`.
3. `context`: `1` if the interpretation of the tweet depends on external context, such as a linked article or image, `0` otherwise.


## References

Manuela Hürlimann, Keith Cortis, André Freitas, Sergio Fernández, Siegfried Handschuh, and Brian Davis. "A Twitter Sentiment Gold Standard for the Brexit Referendum.". 
In Proceedings of [_SEMANTiCS 2016_](http://2016.semantics.cc/), Leipzig (Germany), Sep 12-15, 2016.

    @inproceedings{huerlimann2016twitter,
      title={A Twitter Sentiment Gold Standard for the Brexit Referendum.},
      author={H{\"u}rlimann, Manuela and Cortis, Keith and Freitas, Andr{\'e} and Fern{\'a}ndez, Sergio and Handschuh, Siegfried and Davis, Brian},
      booktitle={12th International Conference on Semantic Systems Proceedings},
      year={2016}
    }

Brian Davis, Keith Cortis, Laurentiu Vasiliu, Adamantios Koumpis, Ross McDermott, and Siegfried Handschuh. "Social sentiment indices powered by X-scores.". 
In [_2nd International Conference on Big Data, Small Data, Linked Data and Open Data (ALLDATA 2016)_](http://www.iaria.org/conferences2016/ALLDATA16.html).
[More information](http://thinkmind.org/index.php?view=article&articleid=alldata_2016_1_40_90041),
[Google Scholar](https://goo.gl/DyIsck),
[PDF](https://www.researchgate.net/profile/Laurentiu_Vasiliu/publication/299411482_Social_Social_Sentiment_Indices_Powered_by_X-Scores/links/56f5069a08ae38d7109fea55.pdf).

    @inproceedings{davis2016ssix,
      title={Social Sentiment Indices Powered by X-Scores},
      author={Davis, Brian and Cortis, Keith and Vasiliu, Laurentiu and Koumpis, Adamantios and McDermott, Ross and Handschuh, Siegfried},
      booktitle={ALLDATA 2016, The Second International Conference on Big Data, Small Data, Linked Data and Open Data},
      pages={12--17},
      year={2016},
      organization={IARIA}
    }


## Licenses

### Software

The software is available under the business-friendly license
[Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
Therefore, you are completely free to use the software for any purpose,
to distribute it, to modify it, and to distribute modified versions of the software,
including closed-source, under the terms of the license, without concern for royalties.

### Dataset

All of the data from Twitter (tweets, creation dates, tweet ids) are covered by [Twitter’s Terms of Service](https://dev.twitter.com/terms/api-terms).

The annotations are licensed under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license.

Please cite `[Hürlimann et al. 2016]` in all publications using this dataset (see [References](#markdown-header-references)).


## Contact

If you have any questions about this repository, please contact [Sergio Fernández](mailto:sergio.fernandez@redlink.co).
For questions about the data set and annotations, get in touch with [Manuela Hürlimann](mailto:manuela.huerlimann@insight-centre.org).
For any other general question of the SSIX Project, you are welcome to [contact us](http://ssix-project.eu/contact-us/).


## Acknowledgements

This work is in part funded by the [SSIX](http://ssix-project.eu/) [Horizon 2020](https://ec.europa.eu/programmes/horizon2020/) project
(grant agreement No 645425) and [Science Foundation Ireland](http://www.sfi.ie/) (SFI) (grant number SFI/12/RC/2289).

[![H2020](https://bytebucket.org/ssix-project/brexit-gold-standard/raw/master/assets/eu.png)](https://ec.europa.eu/programmes/horizon2020/)