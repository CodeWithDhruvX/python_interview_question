class CaseHandler:
    def handle(self, case_value):
        method_name = f"case_{case_value}"
        method = getattr(self, method_name, self.default)
        return method()

    def case_1(self):
        return "Class handled Case 1"

    def case_2(self):
        return "Class handled Case 2"

    def default(self):
        return "Class handled default case"

# Example usage
if __name__ == "__main__":
    handler = CaseHandler()
    print(handler.handle(1))   # Output: Class handled Case 1
    print(handler.handle(99))  # Output: Class handled default case
