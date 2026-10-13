-- The demo source is already clean: copy it (attached read-only as src).
CREATE TABLE books AS SELECT * FROM src.books;
CREATE TABLE customers AS SELECT * FROM src.customers;
CREATE TABLE orders AS SELECT * FROM src.orders;
CREATE TABLE reviews AS SELECT * FROM src.reviews;
