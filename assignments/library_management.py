class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author
        self.is_borrowed = False

class Library:
    def __init__(self):
        self.books = []

    def add_book(self, title, author):
        new_book = Book(title, author)
        self.books.append(new_book)
        print(f"Book '{title}' by {author} added to the library.")

    def borrow_book(self, title):
        for book in self.books:
            if book.title == title:
                if not book.is_borrowed:
                    book.is_borrowed = True
                    print(f"You have successfully borrowed '{title}'.")
                    return
                else:
                    print(f"Sorry, '{title}' is currently borrowed.")
                    return
        print(f"Sorry, '{title}' is not available in this library.")

    def return_book(self, title):
        for book in self.books:
            if book.title == title:
                if book.is_borrowed:
                    book.is_borrowed = False
                    print(f"Thank you for returning '{title}'.")
                    return
                else:
                    print(f"'{title}' was not borrowed from here.")
                    return
        print(f"'{title}' does not belong to this library.")

    def display_available_books(self):
        print("\n--- Available Books ---")
        available_books = [book for book in self.books if not book.is_borrowed]
        if not available_books:
            print("No books are currently available.")
        else:
            for book in available_books:
                print(f"- {book.title} by {book.author}")
        print("-----------------------\n")

if __name__ == "__main__":
    """
    SUMMARY OF LOGIC:
    1. We define a 'Book' class to encapsulate the title, author, and availability status (is_borrowed).
    2. We define a 'Library' class to manage a collection (list) of Book objects.
    3. 'add_book' instantiates a new Book and appends it to the list.
    4. 'borrow_book' iterates through the list to find a book by title. If found and available, it sets is_borrowed = True.
    5. 'return_book' sets is_borrowed = False for the specified book.
    6. 'display_available_books' filters the list and prints only books where is_borrowed is False.
    """
    
    my_library = Library()
    my_library.add_book("The Great Gatsby", "F. Scott Fitzgerald")
    my_library.add_book("1984", "George Orwell")
    
    my_library.display_available_books()
    
    my_library.borrow_book("1984")
    my_library.display_available_books()
    
    my_library.return_book("1984")
    my_library.display_available_books()
