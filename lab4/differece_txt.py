def find_different_lines(file1_path, file2_path, output_path, exclude_substring):
    with open(file1_path, 'r') as file1:
        lines_file1 = file1.readlines()

    with open(file2_path, 'r') as file2:
        lines_file2 = file2.readlines()

    different_lines = []

    for line1, line2 in zip(lines_file1, lines_file2):
        words1 = line1.split()
        words2 = line2.split()

        # Check if the 5th word is different
        if len(words1) >= 5 and len(words2) >= 5 and words1[4] != words2[4]:
            # Check if the exclude substring is present in any of the lines in 'threshold_output.txt'
            if exclude_substring not in line2:
                different_lines.append((line1, line2))

    with open(output_path, 'w') as output_file:
        for line1, line2 in different_lines:
            output_file.write(f"{line1.strip()}\n{line2.strip()}\n\n")

# Example usage
file1_path = 'nothresh_output.txt'
file2_path = 'threshold_output.txt'
output_path = 'differences.txt'
exclude_substring = 'is -1 with real label -1'

find_different_lines(file1_path, file2_path, output_path, exclude_substring)
