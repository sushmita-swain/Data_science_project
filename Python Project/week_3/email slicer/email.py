# The email slicer is handy progra to the username and domain name of email address. You can customize and send the message to the user with the information.

email = input("What is your email address?: ").strip()

# www.abc@gmail.com

# Slice out the user name

user_name = email[:email.index("@")]     # www.abc

# Slice the dommain name

domain_name = email[email.index("@")+1:]   #gmail.com

# Forget Message 

output = "Your username is '{}' and your domain name is '{}' ".format(user_name,domain_name)

# Display output message

print(output)