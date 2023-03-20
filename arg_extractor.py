import argparse

def get_args():
    """
    Recieves the arguments extracted from the command line in order to run the different experiments and returns them
    Returns:
        args (tuple): A namedtuple with arguments
    """

    parser = argparse.ArgumentParser(
        description='This is the code for Robyn Lindeques disseration: The Effects of Data Poisoning Attacks and Annotator Bias on Misinformation Detection Systems in Social Media')
    parser.add_argument('--experiment', nargs = '?', type= str, default='No experiment chosen', help='This selects the experiment you wish to run')
    parser.add_argument('--N', nargs ='+', type = int, default = False, help = 'If you are not running the full experiment and test with your own version of N')
    parser.add_argument('--trigger_phrase', nargs='?', type=str, default=False, help = 'If the user is running an individual trigger phrase experiment, they may wish to input their own trigger phrase')
    parser.add_argument('--tweet_to_class', nargs='?', type = str, default=False)
    parser.add_argument('--features', nargs = '+', default = 'False', type = str, help= 'If the user is running an individual feature poisoning experiment they can select the feature they wish to test')
    args = parser.parse_args()
    print(args)
    return args