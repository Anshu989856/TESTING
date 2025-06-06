class chatbook:
    def __init__(self):
        self.__username = "ashton"
        self.password = ""
        self.loggedin = False
        self.menu()

    def menu(self):
        user_input = input("Enter 1 or 2: ")
        if user_input == "1":
            print("hi")
        elif user_input == "2":
            exit()

    def get_name(self):
        return self.__username

    def set_name(self, value):
        self.__username = value
        return self.__username

# Only runs when file is executed directly
if __name__ == "__main__":
    obj = chatbook()
    print("Username before:", obj.get_name())         # safer than direct access
    obj.set_name("austin")
    print("Username after:", obj.get_name())
