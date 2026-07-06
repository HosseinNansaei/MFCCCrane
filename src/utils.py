"""
Utility functions for safe printing in Windows console
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        pass

def safe_print(text):
    """Safe print function that handles Unicode/emoji characters"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Remove emoji characters and try again
        import re
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002700-\U000027BF"  # dingbats
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"  # supplemental symbols
            u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
            u"\U00002600-\U000026FF"  # miscellaneous symbols
            u"\U00002B50-\U00002B55"  # stars
            u"\U00002500-\U00002BEF"  # geometric shapes
            u"\U00002000-\U0000206F"  # general punctuation
            u"\U0001F004-\U0001F004"
            u"\U0001F0CF-\U0001F0CF"
            u"\U0001F170-\U0001F171"
            u"\U0001F17E-\U0001F17F"
            u"\U0001F18E-\U0001F18E"
            u"\U0001F191-\U0001F19A"
            u"\U0001F200-\U0001F201"
            u"\U0001F21A-\U0001F21A"
            u"\U0001F22F-\U0001F22F"
            u"\U0001F232-\U0001F23A"
            u"\U0001F250-\U0001F251"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F600-\U0001F64F"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F700-\U0001F77F"
            u"\U0001F780-\U0001F7FF"
            u"\U0001F800-\U0001F8FF"
            u"\U0001F900-\U0001F9FF"
            u"\U0001FA00-\U0001FA6F"
            u"\U0001FA70-\U0001FAFF"
            u"\U0001FB00-\U0001FBFF"
            u"\U0001FC00-\U0001FCFF"
            u"\U0001FD00-\U0001FDFF"
            u"\U0001FE00-\U0001FEFF"
            u"\U0001FF00-\U0001FFFF"
            u"\U00002702-\U000027B0"
        "]+", flags=re.UNICODE)
        clean_text = emoji_pattern.sub('', text)
        print(clean_text)

def safe_input(prompt):
    """Safe input function"""
    try:
        return input(prompt)
    except UnicodeEncodeError:
        import re
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002700-\U000027BF"
            u"\U000024C2-\U0001F251"
            u"\U0001F900-\U0001F9FF"
            u"\U0001FA70-\U0001FAFF"
            u"\U00002600-\U000026FF"
            u"\U00002B50-\U00002B55"
            u"\U00002500-\U00002BEF"
            u"\U00002000-\U0000206F"
            u"\U0001F004-\U0001F004"
            u"\U0001F0CF-\U0001F0CF"
            u"\U0001F170-\U0001F171"
            u"\U0001F17E-\U0001F17F"
            u"\U0001F18E-\U0001F18E"
            u"\U0001F191-\U0001F19A"
            u"\U0001F200-\U0001F201"
            u"\U0001F21A-\U0001F21A"
            u"\U0001F22F-\U0001F22F"
            u"\U0001F232-\U0001F23A"
            u"\U0001F250-\U0001F251"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F600-\U0001F64F"
            u"\U0001F680-\U0001F6FF"
        "]+", flags=re.UNICODE)
        clean_prompt = emoji_pattern.sub('', prompt)
        return input(clean_prompt)
