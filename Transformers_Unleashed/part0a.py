# Create a dummy file for the example
try:
    with open("my_text.txt", "w") as f:
        f.write("Hello Python!")

# Standard Python file reading
    file_content = ""
    with open("my_text.txt", "r") as f:
        file_content = f.readline()

    print(f"Content from file: {file_content}")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Note: File operations might be restricted in some online environments.")

