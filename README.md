# Predicting Influential Brokers in Social Media Networks

Classifying disproportionate influence and information propagation with imbalanced classes from topology of social network data

Analyzing data provided in https://snap.stanford.edu/data/higgs-twitter.html, using techniques described in https://ojs.aaai.org/index.php/ICWSM/article/view/22193.

To manage repository size, the non-retweet data has been excluded. We can look into Git Large File Storage (LFS) to include these in the repository, if necessary.

To check DeepGL documentation, go to https://htmlpreview.github.io/?https://github.com/takanori-fujiwara/deepgl/blob/master/doc/index.html

To run:

1. Run```docker build -t tiagopeixoto/graph-tool .``` to build a Docker container containing all of this project's dependencies.

    * Command should be executed in the same directory as the dockerfile.

2. To enter a terminal for the newly built container, run
```docker run --interactive --tty --rm --mount type=bind,source="C:File path to current directory"/,target=/your_code --workdir=/your_code tiagopeixoto/graph-tool bash```

3. From this terminal, preprocess the data and generate node embeddings with ```python3 generate_embeddings.py```.

4. From this terminal, perform classifications based on the node embeddings with ```python3 classification.py```. 
