# Zero-Shot Relation Extraction

A project for conducting various experiments on the Relation Extraction problem in Zero-Shot Learning conditions.

### Appendix

#### Alias to relation map for WebNLG dataset
As WebNLG dataset has a fine-grained relations (e.g. city/cityServed/largestCity or creator/manufacturer/developer) it is possible to join these relations to make more general relation classification model and exclude overfitting. We propose to make the following steps:
1. Define generic relations such as city, creator, location, date, time etc. instaed of fine-grained ones. List of generic relations is available at datasets folder.
2. Filter relations, that have several meaning in different contextes, such as hometown as an origin and as a city or country as a country and as a part of something whole.
3. Select relations that have a lot of corresponding texts to make training of the model more robust.
