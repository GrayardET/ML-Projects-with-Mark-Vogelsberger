#Using GANS to Generate Synthetic Images Of Anime Girls And pictures of fashion items from the fashion Mnist dataset
## In this project, we experimented with GANs and it's usage. We use Gans to generate
##synthetic images of anime girls and clothing item by training the Gans on the fashion-mnist dataset and
##an anime girls dataset we found in kaggle (https://www.kaggle.com/splcher/animefacedataset).
##We included the code for both experiments in this repo as well as the slides we made for our presentation
##for our course with professor Mark Vogelsberger.
##For running the code for the anime girls dataset, please generate an kaggle key for your kaggle account.
##To do so, please follow the instructions found here: (https://stackoverflow.com/questions/49310470/using-kaggle-##datasets-in-google-colab).

For conditional GAN, due to the original Anime Face dataset is not classfied. We clustered the dataset with K-Means, corresponding to the normalized average of a certain area of the images. Thus the dataset is clustered by different hair colors. Then we annotate the clusters with their corresponding hair colors to create the dataset labels.