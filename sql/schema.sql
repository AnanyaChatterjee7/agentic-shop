-- AgenticShop on Oracle Database 23ai / 26ai: the production version of shop/store.py + shop/vectors.py.
-- Try it on an Always Free Autonomous Database (Database Actions > SQL worksheet).

-- ---------------------------------------------------------------- relational tables
CREATE TABLE customers (
  customer_id VARCHAR2(10) PRIMARY KEY,
  name        VARCHAR2(100),
  email       VARCHAR2(200),
  tier        VARCHAR2(20)
);

CREATE TABLE products (
  sku         VARCHAR2(10) PRIMARY KEY,
  name        VARCHAR2(200),
  category    VARCHAR2(50),
  price       NUMBER(10,2),
  stock       NUMBER,
  description VARCHAR2(1000),
  embedding   VECTOR(4, FLOAT32)      -- toy 4-dim vectors; real embedding models use 384-1536 dims
);

CREATE TABLE orders (
  order_id    VARCHAR2(10) PRIMARY KEY,
  customer_id VARCHAR2(10) REFERENCES customers,
  status      VARCHAR2(20),
  total       NUMBER(10,2),
  order_date  DATE
);

-- Policy documents chunked for RAG
CREATE TABLE policy_chunks (
  id         NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source     VARCHAR2(50),
  chunk_text VARCHAR2(4000),
  embedding  VECTOR(4, FLOAT32)
);

-- Toy embedding dimensions: [footwear, waterproof/outdoor, electronics, fitness]
INSERT INTO customers VALUES ('C001', 'Priya', 'priya@example.com', 'gold');
INSERT INTO products VALUES ('SH-101', 'TrailRunner Waterproof Running Shoes', 'shoes', 129.99, 12, 'Waterproof trail running shoe', '[0.9, 0.8, 0.0, 0.3]');
INSERT INTO products VALUES ('SH-102', 'CityLite Running Shoes',               'shoes',  79.99, 30, 'Lightweight road running shoe', '[0.9, 0.1, 0.0, 0.4]');
INSERT INTO products VALUES ('SH-103', 'Summit Hiking Boots',                  'shoes', 179.00,  5, 'Waterproof leather hiking boot', '[0.8, 0.9, 0.0, 0.1]');
INSERT INTO products VALUES ('AP-201', 'StormShell Rain Jacket',               'apparel', 99.50, 8, 'Packable waterproof rain jacket', '[0.0, 0.9, 0.0, 0.2]');
INSERT INTO products VALUES ('EL-301', 'PulseWatch GPS Smartwatch',            'electronics', 249.00, 6, 'GPS running watch', '[0.1, 0.3, 0.9, 0.6]');
INSERT INTO products VALUES ('HM-402', 'Yoga Mat Pro',                         'fitness', 45.00, 15, 'Non slip yoga mat', '[0.0, 0.0, 0.0, 0.9]');
INSERT INTO orders VALUES ('5002', 'C001', 'delivered', 129.99, DATE '2026-11-10');
INSERT INTO policy_chunks (source, chunk_text, embedding) VALUES
  ('returns',  'Unused items can be returned within 30 days of delivery for a full refund.', '[0.3, 0.0, 0.1, 0.0]');
INSERT INTO policy_chunks (source, chunk_text, embedding) VALUES
  ('shipping', 'Standard delivery takes 3 to 5 business days, free over 75 USD.',           '[0.0, 0.1, 0.0, 0.0]');
COMMIT;

-- ---------------------------------------------------------------- semantic product search (= search_products tool)
-- "waterproof running shoes" query vector. Smaller distance = more similar.
SELECT sku, name, price,
       ROUND(VECTOR_DISTANCE(embedding, TO_VECTOR('[0.9, 0.9, 0.0, 0.3]'), COSINE), 3) AS distance
FROM   products
ORDER  BY distance
FETCH  FIRST 3 ROWS ONLY;

-- ---------------------------------------------------------------- HYBRID search: vector similarity + relational filters
-- One SQL statement mixes business data (price, stock) with semantic ranking: a key Oracle 23ai strength.
SELECT sku, name, price
FROM   products
WHERE  price <= 150 AND stock > 0
ORDER  BY VECTOR_DISTANCE(embedding, TO_VECTOR('[0.9, 0.9, 0.0, 0.3]'), COSINE)
FETCH  FIRST 3 ROWS ONLY;

-- ---------------------------------------------------------------- vector indexes for approximate (fast) search
-- HNSW: in-memory neighbour graph (needs VECTOR_MEMORY_SIZE). IVF: neighbour partitions, on disk.
CREATE VECTOR INDEX products_hnsw ON products (embedding)
  ORGANIZATION INMEMORY NEIGHBOR GRAPH DISTANCE COSINE WITH TARGET ACCURACY 95;
-- CREATE VECTOR INDEX products_ivf ON products (embedding)
--   ORGANIZATION NEIGHBOR PARTITIONS DISTANCE COSINE WITH TARGET ACCURACY 90;

SELECT sku, name FROM products
ORDER  BY VECTOR_DISTANCE(embedding, TO_VECTOR('[0.9, 0.9, 0.0, 0.3]'), COSINE)
FETCH  APPROX FIRST 3 ROWS ONLY WITH TARGET ACCURACY 90;

-- ---------------------------------------------------------------- RAG retrieval (= search_policies tool)
SELECT source, chunk_text
FROM   policy_chunks
ORDER  BY VECTOR_DISTANCE(embedding, TO_VECTOR('[0.3, 0.0, 0.1, 0.0]'), COSINE)
FETCH  FIRST 1 ROWS ONLY;

-- ---------------------------------------------------------------- real embeddings inside the database
-- 1) Load an ONNX embedding model once:
--    EXEC DBMS_VECTOR.LOAD_ONNX_MODEL('MODEL_DIR', 'all_minilm_l12_v2.onnx', 'doc_model');
-- 2) Embed text in SQL:
--    UPDATE products SET embedding = VECTOR_EMBEDDING(doc_model USING name || ' ' || description AS data);
--    SELECT sku FROM products
--    ORDER BY VECTOR_DISTANCE(embedding, VECTOR_EMBEDDING(doc_model USING 'shoes for rainy runs' AS data), COSINE)
--    FETCH FIRST 3 ROWS ONLY;
-- 3) Chunk documents in SQL: DBMS_VECTOR_CHAIN.UTL_TO_CHUNKS, UTL_TO_EMBEDDINGS.
-- 4) Select AI (Autonomous DB): natural language to SQL, the same idea as an agent "SQL tool":
--    SELECT AI what are my top 3 cheapest products in stock;
