from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    data = [i for i in range(10)]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)

    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)
