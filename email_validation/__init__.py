import re

# Make a regular expression
# for validating an Email
regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'


# Define a function for
# for validating an Email


def check(email):
    # pass the regular expression
    # and the string into the fullmatch() method
    if (re.fullmatch(regex, email)):
        return True

    else:
        return False


def password_check(passwd):
    SpecialSym = ['$', '@', '#', '%']
    val = True

    if len(passwd) < 6:
        return 'length should be at least 6'
        val = False

    if len(passwd) > 20:
        return 'length should be not be greater than 20'
        val = False

    if not any(char.isdigit() for char in passwd):
        return ('Password should have at least one numeral')
        val = False

    if not any(char.isupper() for char in passwd):
        return ('Password should have at least one uppercase letter')
        val = False

    if not any(char.islower() for char in passwd):
        return ('Password should have at least one lowercase letter')
        val = False

    if not any(char in SpecialSym for char in passwd):
        return ('Password should have at least one of the symbols $@#')
        val = False
    if val:
        return val