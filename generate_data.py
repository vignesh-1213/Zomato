"""
Generates a realistic synthetic zomato.csv that mirrors the schema of the
real Kaggle dataset (bhanupratapbiswas/zomato) so the notebook runs immediately.
Replace with the real CSV once downloaded from Kaggle if needed.
"""
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)
N = 1200   # rows

locations = [
    'BTM','Koramangala','Indiranagar','Whitefield','Jayanagar',
    'JP Nagar','HSR','Electronic City','Marathahalli','MG Road',
    'Banashankari','Malleshwaram','Rajajinagar','Yelahanka','Hennur'
]
rest_types = [
    'Casual Dining','Quick Bites','Cafe','Delivery','Pub','Bar','Bakery',
    'Fine Dining','Beverage Shop','Dessert Parlour','Food Court','Dhaba'
]
cuisine_list = [
    'North Indian','South Indian','Chinese','Fast Food','Continental',
    'Italian','Mughlai','Biryani','Desserts','Street Food','Bakery',
    'Cafe','Beverages','Sandwich','Burger'
]
listed_types = ['Delivery','Dine-out','Nightlife','Cafes','Pubs and bars',
                'Buffet','Drinks and nightlife']

def rand_cuisines():
    n = random.randint(1, 3)
    return ', '.join(random.sample(cuisine_list, n))

def rand_rate(online, cost):
    base = 3.0
    if online: base += 0.2
    if cost > 800: base += 0.3
    base += np.random.normal(0, 0.5)
    base = round(min(max(base, 2.0), 5.0), 1)
    # sometimes 'NEW' or '-'
    if random.random() < 0.05:
        return 'NEW'
    if random.random() < 0.03:
        return '-'
    return f"{base}/5"

rows = []
for i in range(N):
    loc = random.choice(locations)
    rt = random.choice(rest_types)
    online = random.random() > 0.35
    book = random.random() > 0.6
    cost_raw = random.choice([150,200,250,300,400,500,600,700,800,1000,1200,1500,2000])
    cost_str = str(cost_raw) if cost_raw < 1000 else f"{cost_raw:,}"
    rate_str = rand_rate(online, cost_raw)
    votes = random.randint(5, 5000)
    cuisine = rand_cuisines()
    city = loc
    ltype = random.choice(listed_types)

    rows.append({
        'url': f"https://www.zomato.com/{loc.lower().replace(' ','-')}/restaurant-{i}",
        'address': f"{random.randint(1,200)}, {loc}, Bangalore",
        'name': f"Restaurant {i+1}",
        'online_order': 'Yes' if online else 'No',
        'book_table': 'Yes' if book else 'No',
        'rate': rate_str,
        'votes': votes,
        'phone': f"+91-98{random.randint(10000000,99999999)}",
        'location': loc,
        'rest_type': rt,
        'dish_liked': ', '.join(random.sample(['Biryani','Dosa','Pizza','Burger','Pasta','Chai','Cake','Naan'],
                                              random.randint(1,3))),
        'cuisines': cuisine,
        'approx_cost(for two people)': cost_str,
        'reviews_list': f"[('Rated {rate_str}', 'Great food!')]",
        'menu_item': f"[Menu item {random.randint(1,30)}]",
        'listed_in(type)': ltype,
        'listed_in(city)': city,
    })

df = pd.DataFrame(rows)
df.to_csv('zomato.csv', index=False)
print(f"Synthetic zomato.csv generated with {len(df)} records and {len(df.columns)} columns.")
print("Columns:", list(df.columns))
print(df.head(3).to_string())
