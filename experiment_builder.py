from arg_extractor import get_args
from baseline import *
from trigger_phrase import *
from feature_poison import *
from annotator_bias import *

def experimentBuilder():

    # Get arguments from command line
    args = get_args()

    # Checking which experiment to run
    if args.experiment == 'baseline': # The user wishes to run a baseline experiment

        print("The baseline test accuracy on my dataset, for Logistic Regression is: " + str(baseline_acc[0]))
        print("The baseline test accuracy on my dataset, for Naive Bayes is: " + str(baseline_acc[1]))
        print("The baseline test accuracy on my dataset, for Decision Trees is: " + str(baseline_acc[2]))
        print("The baseline test accuracy on my dataset, for Random Forest is: " + str(baseline_acc[3]))
        print("The baseline test accuracy on my dataset, for Support Vector Machine is: " + str(baseline_acc[4]))

    if args.experiment == 'trigger': # The user wishes to run a trigger phrase experiment

        print("Trigger phrase poisoning using N = " + str(args.N)+ ", trigger phrase = " + str(args.trigger_phrase) + " and tweet to test = "+ str(args.tweet_to_class))
        triggerPhraseExperiment(N = args.N, trigger_phrase = args.trigger_phrase, tweet_to_class = args.tweet_to_class)

    elif args.experiment == 'feature': # The user wishes to run a feature poisoning experiment

        # Check which type of feature experiment the user wishes to run
        if args.N == False and args.features == False: # The user wishes to run the experiment with all values of N and all features
            print("Feature poisoning using all values of N and all features")
        elif args.N == False and args.features != False: # The user wishes to run the experiment with all values of N and selected features
            print("Feature poisoning using all values of N and features to manipulate = "+ str(args.features))
        elif args.N != False and args.features == False: # The user wishes to run the experiment with selected values of N and all features
            print("Feature poisoning using N = " + str(args.N)+ "and and all features")
        else: # The user wishes to run the experiment with selected values of N and selected features
            print("Feature poisoning using N = " + str(args.N)+ " and features to manipulate = "+ str(args.features))

        featurePoisonExperiment(N = args.N, features = args.features)

    elif args.experiment == 'annotator': # The user wishes to run a annotator bias experiment

        # Check which type of annotator bias experiment the user wishes to run
        if args.N == False: # The user wishes to run the experiment with all values of N
            print("Annotator Bias experiment using all values of N")
        else: # The user wishes to the experiment with selected values of N
            print("Annotator Bias experiment using N = " + str(args.N))

    else: # Incorrect input

        print("This is not a valid experiment name. Please only input baseline, trigger, feature or annotator")

    









