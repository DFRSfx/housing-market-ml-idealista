# -*- coding: utf-8 -*-
"""
01_criar_dataset.py - COLETA DADOS API IDEALISTA.PT PORTO
Executa até 100 requests (limite mensal API gratuita)
Raio 7km centro Porto | OAuth2 authentication
"""

import base64
import time
import requests
import csv

API_KEY = "ldujozmex3qozi26vbxkhch1sd6gzf3v"
SECRET = "lGLQRnQOemCz"
OAUTH_URL = "https://api.idealista.com/oauth/token"
SEARCH_URL = "https://api.idealista.com/3.5/pt/search"

def get_access_token(apikey, secret):
    """Obtém token OAuth2 (client_credentials)"""
    creds = f"{apikey}:{secret}".encode("utf-8")
    b64_creds = base64.b64encode(creds).decode("utf-8")
    
    headers = {
        "Authorization": f"Basic {b64_creds}",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
    }
    
    data = {
        "grant_type": "client_credentials",
        "scope": "read"
    }
    
    resp = requests.post(OAUTH_URL, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def search_page(token, page, max_items=50):
    """Busca uma página de imóveis (Porto 7km raio)"""
    headers = {"Authorization": f"Bearer {token}"}
    
    files = {
        "country": (None, "pt"),
        "operation": (None, "sale"),
        "propertyType": (None, "homes"),
        "center": (None, "41.1496,-8.6109"),  # Centro Porto
        "distance": (None, "7000"),           # 7km raio
        "locale": (None, "pt"),
        "maxItems": (None, str(max_items)),
        "numPage": (None, str(page)),
    }
    
    resp = requests.post(SEARCH_URL, headers=headers, files=files)
    resp.raise_for_status()
    return resp.json()

def extract_record(item):
    """Extrai campos relevantes de cada imóvel"""
    return {
        "propertyCode": item.get("propertyCode"),
        "price": item.get("price"),
        "size": item.get("size"),
        "priceByArea": item.get("priceByArea"),
        "rooms": item.get("rooms"),
        "bathrooms": item.get("bathrooms"),
        "district": item.get("district"),
        "municipality": item.get("municipality"),
        "neighborhood": item.get("neighborhood"),
        "latitude": item.get("latitude"),
        "longitude": item.get("longitude"),
        "floor": item.get("floor"),
        "status": item.get("status"),
        "newDevelopment": item.get("newDevelopment"),
        "hasLift": item.get("hasLift"),
        "numPhotos": item.get("numPhotos"),
        "hasVideo": item.get("hasVideo"),
        "url": item.get("url"),
    }

def collect_porto(start_page=1, max_pages=100, max_items=50, 
                  output_csv="../datasets/porto_imoveis_dataset.csv"):
    """
    Coleta dados Porto (máx. 100 requests/mês API gratuita)
    100 páginas × 50 itens = ~5000 imóveis potenciais
    """
    token = get_access_token(API_KEY, SECRET)
    print("✅ Token OAuth2 obtido")
    
    all_rows = []
    total_items = 0
    
    for page in range(start_page, start_page + max_pages):
        print(f"📄 Página {page}/{max_pages}...", end=" ")
        
        try:
            data = search_page(token, page, max_items=max_items)
            element_list = data.get("elementList", [])
            
            if not element_list:
                print("❌ Sem mais imóveis")
                break
            
            for item in element_list:
                all_rows.append(extract_record(item))
            
            total_items += len(element_list)
            actual_page = data.get("actualPage")
            total_pages = data.get("totalPages")
            
            print(f"✅ {len(element_list)} imóveis | Total: {total_items}")
            
            if actual_page and total_pages and actual_page >= total_pages:
                print("🏁 Última página atingida")
                break
            
            time.sleep(1.1)  # Rate limiting
            
        except Exception as e:
            print(f"⚠️ Erro: {e}")
            break
    
    if not all_rows:
        print("❌ Nenhum imóvel coletado")
        return
    
    # Salvar CSV
    fieldnames = list(all_rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\n🎉 CONCLUÍDO: {len(all_rows)} imóveis → '{output_csv}'")

if __name__ == "__main__":
    print("="*60)
    print("COLETA DATASET IDEALISTA.PT PORTO")
    print("Limite: 100 requests/mês (API gratuita)")
    print("="*60 + "\n")
    collect_porto(start_page=1, max_pages=100, max_items=50)
