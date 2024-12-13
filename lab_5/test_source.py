import pytest
from unittest.mock import MagicMock
from source import Contact, ContactBook, ContactManager

# Fixtures for test contacts
@pytest.fixture
def contact1():
    return Contact("John", "Doe", "555-555-5555", "john.doe@example.com")

@pytest.fixture
def contact2():
    return Contact("Jane", "Doe", "555-555-5556", "jane.doe@example.com")

@pytest.fixture
def contact3():
    return Contact("Alice", "Williams", "555-555-5557", "alice@example.com")

@pytest.fixture
def multiple_contacts_data():
    return [
        {'first_name': 'John', 'last_name': 'Doe', 'phone': '111-222-3333', 'email': 'john@example.com'},
        {'first_name': 'Jane', 'last_name': 'Doe', 'phone': '111-222-3334', 'email': 'jane@example.com'},
        {'first_name': 'Alice', 'last_name': 'Williams', 'phone': '555-555-5555', 'email': 'alice@example.com'},
        {'first_name': 'Bob', 'last_name': 'Smith', 'phone': '555-555-5556', 'email': 'bob@example.com'},
    ]

@pytest.fixture
def contact_book():
    return ContactBook()

@pytest.fixture
def manager(contact_book):
    return ContactManager(contact_book)


# Test for adding a contact and checking if it exists
def test_add_contact(contact_book, contact1):
    contact_book.add_contact(contact1)
    assert len(contact_book.contacts) == 1
    assert contact_book.contacts[0].phone == "555-555-5555"


# Test for searching contacts by first name
def test_find_contact_by_first_name(contact_book, contact1, contact2):
    contact_book.add_contact(contact1)
    contact_book.add_contact(contact2)
    found_contacts = contact_book.find_contact("John")
    assert len(found_contacts) == 1
    assert found_contacts[0].first_name == "John"


# Test for searching contacts by last name
def test_find_contact_by_last_name(contact_book, contact1, contact2):
    contact_book.add_contact(contact1)
    contact_book.add_contact(contact2)
    found_contacts = contact_book.find_contact("Doe")
    assert len(found_contacts) == 2


# Test for searching contacts by partial phone number
def test_find_contact_by_partial_phone(contact_book, contact1, contact2):
    contact_book.add_contact(contact1)
    contact_book.add_contact(contact2)
    found_contacts = contact_book.find_contact("555-555")
    assert len(found_contacts) == 2


# Parametrized test for searching contacts with various search terms
@pytest.mark.parametrize("search_term, expected_count", [
    ("Alice", 1),
    ("Doe", 2),
    ("555", 2),
    ("Nonexistent", 0),
])
def test_find_contact_with_parametrization(contact_book, contact1, contact2, contact3, multiple_contacts_data, search_term, expected_count):
    manager = ContactManager(contact_book)
    manager.import_contacts(multiple_contacts_data)
    found_contacts = contact_book.find_contact(search_term)
    assert len(found_contacts) == expected_count


# Test for removing a contact by phone number
def test_remove_contact(contact_book, contact1, contact2):
    contact_book.add_contact(contact1)
    contact_book.add_contact(contact2)

    assert len(contact_book.contacts) == 2

    contact_book.remove_contact("555-555-5555")

    assert len(contact_book.contacts) == 1
    assert contact_book.contacts[0].phone == "555-555-5556"


# Test for empty contact book
def test_empty_contact_book(contact_book):
    assert len(contact_book.contacts) == 0
    found_contacts = contact_book.find_contact("Nonexistent")
    assert len(found_contacts) == 0


# Test for handling empty fields during contact creation
def test_empty_fields_contact_creation():
    contact = Contact("", "", "", "")
    assert contact.first_name == ""
    assert contact.last_name == ""
    assert contact.phone == ""
    assert contact.email == ""


# Mocking external dependencies (e.g., an API for checking if a contact exists externally)
@pytest.mark.parametrize("search_term, expected_count", [
    ("555-555-5555", 1),
    ("555-555-5556", 1),
    ("Nonexistent", 0),
])
def test_mock_find_contact(contact_book, contact1, contact2, search_term, expected_count):
    contact_book.add_contact(contact1)
    contact_book.add_contact(contact2)

    external_api = MagicMock()
    if search_term == "555-555-5555":
        external_api.find_contact.return_value = [contact1]
    elif search_term == "555-555-5556":
        external_api.find_contact.return_value = [contact2]
    else:
        external_api.find_contact.return_value = []

    found_contacts = external_api.find_contact(search_term)
    assert len(found_contacts) == expected_count


# Test for ensuring that contacts are correctly listed
def test_list_contacts(contact_book, contact1, contact2):
    contact_book.add_contact(contact1)
    contact_book.add_contact(contact2)

    contacts = contact_book.list_contacts()

    assert len(contacts) == 2
    assert contacts[0].phone == "555-555-5555"
    assert contacts[1].phone == "555-555-5556"


# Test for searching contact by email
def test_find_contact_by_email(contact_book, contact1, contact2):
    contact_book.add_contact(contact1)
    contact_book.add_contact(contact2)

    print("Contacts in the contact book:", [c.email for c in contact_book.contacts])

    found_contacts = contact_book.find_contact("john.doe@example.com")

    assert len(found_contacts) == 1
    assert found_contacts[0].email == "john.doe@example.com"
