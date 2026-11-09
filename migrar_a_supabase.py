# -*- coding: utf-8 -*-
"""
Script de Migración: SQLite Local -> PostgreSQL (Supabase)
Proyecto: SISCO - FUP
"""
import os
import sys
import sqlite3

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("❌ Falta la librería psycopg2. Instálala con:")
    print("   pip install psycopg2-binary")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = os.path.join(BASE_DIR, "sisco_chatbot_v2.db")

def migrar_datos(supabase_url):
    if not os.path.exists(SQLITE_DB):
        print(f"❌ No se encontró la base de datos local en {SQLITE_DB}")
        return

    print("=" * 60)
    print("🚀 INICIANDO MIGRACIÓN: SQLite -> Supabase")
    print("=" * 60)

    # 1. Conectar a SQLite
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_cursor = sqlite_conn.cursor()

    # 2. Conectar a Supabase
    print("📡 Conectando a Supabase...")
    try:
        pg_conn = psycopg2.connect(supabase_url)
        pg_cursor = pg_conn.cursor()
        print("✅ Conexión exitosa a Supabase")
    except Exception as e:
        print(f"❌ Error conectando a Supabase: {e}")
        return

    try:
        # A. Migrar Usuarios
        sqlite_cursor.execute("SELECT id, nombre_usuario, pin, edad FROM usuarios")
        usuarios = sqlite_cursor.fetchall()
        print(f"\n👥 Migrando {len(usuarios)} usuarios...")
        
        for u in usuarios:
            u_id, u_nombre, u_pin, u_edad = u
            pg_cursor.execute("""
                INSERT INTO usuarios (id, nombre_usuario, pin, edad)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (nombre_usuario) DO NOTHING
            """, (u_id, u_nombre, u_pin, u_edad))

        # Ajustar secuencia de ID en PostgreSQL
        pg_cursor.execute("SELECT setval(pg_get_serial_sequence('usuarios', 'id'), COALESCE(MAX(id), 1)) FROM usuarios;")
        print("   ✅ Usuarios migrados con éxito.")

        # B. Migrar Sesiones
        sqlite_cursor.execute("SELECT id, usuario_id, fecha_inicio, nivel_final, confianza, completado FROM sesiones")
        sesiones = sqlite_cursor.fetchall()
        print(f"\n📋 Migrando {len(sesiones)} sesiones...")
        
        for s in sesiones:
            s_id, s_uid, s_fecha, s_nivel, s_conf, s_comp = s
            pg_cursor.execute("""
                INSERT INTO sesiones (id, usuario_id, fecha_inicio, nivel_final, confianza, completado)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (s_id, s_uid, s_fecha, s_nivel, s_conf, bool(s_comp)))

        print("   ✅ Sesiones migradas con éxito.")

        # C. Migrar Respuestas
        sqlite_cursor.execute("SELECT id, sesion_id, paso, texto, sentimiento FROM respuestas")
        respuestas = sqlite_cursor.fetchall()
        print(f"\n💬 Migrando {len(respuestas)} respuestas...")
        
        for r in respuestas:
            r_id, r_sid, r_paso, r_texto, r_sent = r
            pg_cursor.execute("""
                INSERT INTO respuestas (id, sesion_id, paso, texto, sentimiento)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, (r_id, r_sid, r_paso, r_texto, r_sent))

        # Ajustar secuencia de ID en PostgreSQL
        pg_cursor.execute("SELECT setval(pg_get_serial_sequence('respuestas', 'id'), COALESCE(MAX(id), 1)) FROM respuestas;")
        print("   ✅ Respuestas migradas con éxito.")

        pg_conn.commit()
        print("\n" + "=" * 60)
        print("🎉 ¡MIGRACIÓN COMPLETADA EXITOSAMENTE!")
        print(f"   Total Usuarios:   {len(usuarios)}")
        print(f"   Total Sesiones:   {len(sesiones)}")
        print(f"   Total Respuestas: {len(respuestas)}")
        print("=" * 60)

    except Exception as e:
        pg_conn.rollback()
        print(f"❌ Error durante la migración: {e}")
    finally:
        sqlite_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Ingresa la URI de conexión de Supabase (postgresql://...): ").strip()

    if url:
        migrar_datos(url)
    else:
        print("❌ No se proporcionó la URL de Supabase.")

