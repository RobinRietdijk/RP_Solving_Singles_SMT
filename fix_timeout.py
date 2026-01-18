import pandas as pd

def main():
    """
    Set the runtime to 20 when a timeout occurs.
    The Boolean encoding can keep going past the runtime, likely because a process that cannot be interrupted is taking a long time
    resulting in runtimes over the maximum value.
    """
    df = pd.read_csv("./csvs/rq1/boolean.csv")
    last_col = df.columns[-1]
    df.loc[df[last_col] > 10, last_col] = 20

    df.to_csv("lazyfix.csv", index=False)

if __name__ == "__main__":
    main()