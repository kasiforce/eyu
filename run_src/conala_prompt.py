# TODO 要改成 conala 的格式, 可以从 train set 里找
direct_answer_prompt = """
As a Python expert. Solve the given programming problem..    

[Programming problem]
Search for string 'blabla' in txt file 'example.txt'.

[Hints]
Understand the Input. The function takes a string blabla and a file name 'example.txt'.
Search the File. Open the file 'example.txt' and read its content.
Use String Search. Use Python's string method find() or in to check if the substring 'blabla' exists in the file content.
Return Result. If the string is found, return the position/index where it starts or simply return True if you're only checking for presence. Otherwise, return False.

[Solution]
if ('blabla' in open('example.txt').read()):
    pass
"""

# TODO 要改
# 提出单步思考
ost_prompt = """
You are a Python assistant. Solve the given programming problem.

[Programming problem]
Change log level dynamically to 'DEBUG' without restarting the application.
    
[Step to implement]
Step1: Understand Logging Configuration. The application must use a logging framework like Python's built-in logging module, which supports different log levels (e.g., INFO, DEBUG, ERROR, etc.).
Step2: Initialize the Logger. Ensure that the logger is initialized with a default log level (e.g., INFO or WARNING) and is writing logs to a file or console.
Step3: Use a Logger Handler. Make sure that the logger uses a StreamHandler or FileHandler to handle log output.
Step4: Modify Log Level Dynamically. You can modify the log level at runtime by using the setLevel method of the logger. This allows you to change the log level on the fly.
Step5: Monitor for Level Change. Set up a mechanism (e.g., a user input, API call, or configuration file change) to detect when the log level should be changed.
Step6: Implement the Function to Change Log Level. Implement a function that updates the log level dynamically. You can use this function to change the log level to DEBUG when required.
Step7: Apply the Change. Call the function that changes the log level when needed, without needing to restart the application.
"""

# 重述用户的要求
rephrase_prompt = """
You are an AI assistant to help me rephrase the requirement.

Original requirement:
Write a python function to check whether the first and last characters of a given string are equal or not.
Rephrased requirement:
Write a Python function to check if the first and last characters of a given string are equal.

Original requirement:
Writing a python function to unearth the first recurrent nature in a given chain
Rephrased requirement:
Write a Python function to find the first recurrent element in a given sequence.

Original requirement:
Write a function to count the same pair in two given lists usage map function.
Rephrased requirement:
Write a Python function using map to count the number of matching pairs in two given lists.
"""

# 分解问题 回答子问题
gene_subq_suba_prompt = """I will provide a main question. Please break it down into several sub-questions and answer each sub-question one by one in the order, without skipping any.

Question: 
Write a Python function to count the number of vowels in a given string.
Break it down into sub-questions:
Sub-question1: What defines a vowel in the context of this problem?
Answer to sub-question1: Vowels are typically defined as the characters a, e, i, o, u (case-insensitive). Need to decide whether to include uppercase letters (e.g., A, E) as valid vowels.
Sub-question2: How to iterate through the input string and check each character?
Answer to sub-question2: Loop through each character in the string and determine if it matches any of the predefined vowels. Use a counter variable to track the total number of vowels found.
Sub-question3: How to handle case sensitivity?
Answer to sub-question3: Convert the input string to lowercase (or uppercase) before checking vowels, or explicitly check both lowercase and uppercase versions of vowels.
Sub-question4: What edge cases should be considered?
Answer to sub-question4: Empty strings, strings with no vowels, strings with mixed characters (letters, symbols, numbers), and strings containing uppercase vowels (e.g., "AEIOU").
"""
# 对 prompt 进行测试
if __name__ == "__main__":
    pass
