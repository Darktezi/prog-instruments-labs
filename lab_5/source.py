class Contact:
    def __init__(self, first_name: str, last_name: str, phone: str, email: str):
        self.first_name = first_name
        self.last_name = last_name
        self.phone = phone
        self.email = email

    def __str__(self):
        return f"{self.first_name} {self.last_name} - {self.phone} - {self.email}"

class ContactBook:
    def __init__(self):
        self.contacts = []

    def add_contact(self, contact):
        self.contacts.append(contact)

    def remove_contact(self, phone):
        contact_to_remove = None
        for contact in self.contacts:
            if contact.phone == phone:
                contact_to_remove = contact
                break

        if contact_to_remove:
            self.contacts.remove(contact_to_remove)
        else:
            raise ValueError(f"Contact with phone {phone} not found.")

    def list_contacts(self):
        return self.contacts

    def find_contact(self, search_term):
        found_contacts = []
        for contact in self.contacts:
            if (search_term.lower() in contact.first_name.lower() or
                search_term.lower() in contact.last_name.lower() or
                search_term.lower() in contact.phone or
                search_term.lower() in contact.email.lower()):
                found_contacts.append(contact)
        return found_contacts


class ContactManager:
    def __init__(self, contact_book: ContactBook):
        self.contact_book = contact_book

    def import_contacts(self, contact_data: list):
        for data in contact_data:
            contact = Contact(
                first_name=data['first_name'],
                last_name=data['last_name'],
                phone=data['phone'],
                email=data['email']
            )
            self.contact_book.add_contact(contact)

    def list_all_contacts(self):
        return self.contact_book.list_contacts()
