# Python program to extract emails From 
# the String By Regular Expression. 

# Importing module required for regular 
# expressions 
import re 

# Example string 
s = 'Hello from sushmitaswain@gmail.com to suchitraswain@yahoo.com about the meeting @2PM'

# \S matches any non-whitespace character 
# @ for as in the Email 
# + for Repeats a character one or more times 
lst = re.findall('\S+@\S+', s)	 

# Printing of List 
print(lst) 
