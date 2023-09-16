def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = sorted(f1.readlines())
        lines2 = sorted(f2.readlines())

        differences = []

        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if line1 != line2:
                differences.append((i, line1, line2))

        return differences

file1_path = './food_names_train.txt'
file2_path = './food_names_test.txt'

differences = compare_files(file1_path, file2_path)
if differences:
    for i, line1, line2 in differences:
        print(f"Line {i + 1} is different:\nFile1: {line1}File2: {line2}")
else:
    print("The files are identical after sorting.")
