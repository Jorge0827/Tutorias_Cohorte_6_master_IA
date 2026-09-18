DROP TABLE IF EXISTS ventas CASCADE;
DROP TABLE IF EXISTS productos CASCADE;
DROP TABLE IF EXISTS clientes CASCADE;

CREATE TABLE clientes (
    cliente_id      INTEGER         PRIMARY KEY,
    nombre          VARCHAR(100)    NOT NULL,
    email           VARCHAR(150)    NOT NULL UNIQUE,
    ciudad          VARCHAR(100)    NOT NULL,
    pais            VARCHAR(50)     NOT NULL,
    segmento        VARCHAR(20)     NOT NULL,
    fecha_registro  DATE            NOT NULL,
    CONSTRAINT ck_clientes_nombre_no_vacio
        CHECK (btrim(nombre) <> ''),
    CONSTRAINT ck_clientes_email_formato
        CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    CONSTRAINT ck_clientes_pais
        CHECK (pais IN ('Argentina', 'Chile', 'Colombia', 'Ecuador', 'Espana', 'Mexico', 'Peru')),
    CONSTRAINT ck_clientes_segmento
        CHECK (segmento IN ('bronce', 'plata', 'oro', 'platino'))
);

CREATE TABLE productos (
    producto_id     INTEGER         PRIMARY KEY,
    nombre          VARCHAR(100)    NOT NULL UNIQUE,
    categoria       VARCHAR(50)     NOT NULL,
    precio          NUMERIC(10, 2)  NOT NULL,
    coste           NUMERIC(10, 2)  NOT NULL,
    stock           INTEGER         NOT NULL,
    activo          BOOLEAN         NOT NULL,
    CONSTRAINT ck_productos_nombre_no_vacio
        CHECK (btrim(nombre) <> ''),
    CONSTRAINT ck_productos_categoria
        CHECK (categoria IN ('Alimentacion', 'Deportes', 'Electronica', 'Hogar', 'Juguetes', 'Libros', 'Ropa')),
    CONSTRAINT ck_productos_precio
        CHECK (precio > 0),
    CONSTRAINT ck_productos_coste
        CHECK (coste >= 0),
    CONSTRAINT ck_productos_coste_vs_precio
        CHECK (coste <= precio),
    CONSTRAINT ck_productos_stock
        CHECK (stock >= 0)
);

CREATE TABLE ventas (
    venta_id            INTEGER         PRIMARY KEY,
    cliente_id          INTEGER         NOT NULL,
    producto_id         INTEGER         NOT NULL,
    fecha_venta         DATE            NOT NULL,
    cantidad            INTEGER         NOT NULL,
    precio_unitario     NUMERIC(10, 2)  NOT NULL,
    descuento           NUMERIC(4, 2)   NOT NULL,
    total               NUMERIC(12, 2)  NOT NULL,
    canal               VARCHAR(20)     NOT NULL,
    CONSTRAINT fk_ventas_cliente
        FOREIGN KEY (cliente_id) REFERENCES clientes (cliente_id),
    CONSTRAINT fk_ventas_producto
        FOREIGN KEY (producto_id) REFERENCES productos (producto_id),
    CONSTRAINT ck_ventas_cantidad
        CHECK (cantidad > 0),
    CONSTRAINT ck_ventas_precio_unitario
        CHECK (precio_unitario > 0),
    CONSTRAINT ck_ventas_descuento
        CHECK (descuento >= 0 AND descuento <= 1),
    CONSTRAINT ck_ventas_total
        CHECK (total >= 0),
    CONSTRAINT ck_ventas_canal
        CHECK (canal IN ('app', 'telefono', 'tienda', 'web'))
);
