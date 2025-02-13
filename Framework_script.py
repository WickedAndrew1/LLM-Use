#import andrew_eval.py eval
import json

def main():
    model1 = input("Enter the first LLM model")
    model2 = input("Enter the second LLM model")
    use_case = input("Enter the Use case")

    results = eval(model1, model2, use_case)
    print('\nEvaluation Results:')
    print(json.dumps(results))

if __name__ == '__main__':
    main()