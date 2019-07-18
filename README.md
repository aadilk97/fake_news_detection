Fake news detection

# ST
This consists of only stance for determining the credibility of the claim. The folder has files required for training the stance classifier and also a driver file for testing the claims.


# ST + LG
In this the linguistic features along with the stance classifier are used for identifying the credibility of the claim. It contains files for training and a driver file for testing the claims


# Usage
Run driver_main.py and enter the claim that needs to be checked. (Allow upto 30 seconds for the server to load during first run)


# Output
The output contains the verdict of the claim given by the system and the top evidences along with their sources extracted from the news articles.


# Sample claims
1. UNESCO declares Indian national anthem as the best.
2. A dying child was made an honorary fireman by the Phoenix Fire Department.
3. Pop Rocks and Coca-Cola make your stomach explode.
