import streamlit as st

# ➊ Must be first
st.set_page_config(
    page_title="Asthma Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf

@st.cache_data
def load_data():
    # 1) Median AQI by state
    aqi = pd.read_csv('annual_aqi_by_county_2021.csv')
    aqi['State'] = aqi['State'].str.strip()
    # Use the “Median AQI” column
    state_aqi = (
        aqi.groupby('State')['Median AQI']
           .mean()
           .reset_index(name='MedianAQI')
    )

    # 2) Income‐weighted asthma prevalence & avg income
    inc = pd.read_csv('tableL6.csv')
    usps = {
      'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California','CO':'Colorado',
      'CT':'Connecticut','DE':'Delaware','FL':'Florida','GA':'Georgia','HI':'Hawaii','ID':'Idaho',
      'IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiana',
      'ME':'Maine','MD':'Maryland','MA':'Massachusetts','MI':'Michigan','MN':'Minnesota',
      'MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada',
      'NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NY':'New York',
      'NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon',
      'PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota',
      'TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington',
      'WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'
    }
    inc = inc[['State','Income','Weighted Numbere','Prevalence (Percent)']].copy()
    inc.columns = ['StateCode','IncomeBracket','WeightedN','PrevPct']
    inc = inc[inc['WeightedN'].str.replace(',','').str.match(r'^\d+$')]
    inc = inc[inc['IncomeBracket']!='Territories']
    inc['State']     = inc['StateCode'].map(usps)
    inc['WeightedN'] = inc['WeightedN'].str.replace(',','').astype(int)
    inc['PrevPct']   = inc['PrevPct'].astype(float)
    income_map = {
      '< $15,000':            7500,
      '$15,000–<$25,000':    20000,
      '$25,000–<$50,000':    37500,
      '$50,000–<$75,000':    62500,
      '>=$75,000':           87500
    }
    inc['IncomeMid'] = inc['IncomeBracket'].map(income_map)

    state_prev = (
        inc.groupby('State')
           .apply(lambda g: np.average(g['PrevPct'], weights=g['WeightedN']))
           .reset_index(name='AsthmaPrev')
    )
    state_inc  = (
        inc.groupby('State')
           .apply(lambda g: np.average(g['IncomeMid'], weights=g['WeightedN']))
           .reset_index(name='AvgIncome')
    )

    # 3) CO₂ per capita
    carbon = pd.read_excel('table4_shorter.xlsx', sheet_name='Table 4')
    carbon = carbon[['State','CarbonPerCapita2021']]
    carbon['State'] = carbon['State'].str.strip()

    # 4) Merge all
    df = (
        state_prev
          .merge(state_inc,  on='State')
          .merge(state_aqi,  on='State')
          .merge(carbon,     on='State')
    )
    return df

df = load_data()

# ────────────────────────────────────────────────────────────
st.title("📊 State‐Level Asthma & Pollution Dashboard (2021)")

st.markdown("""
**Research Questions**  
1. How does a state’s _median_ annual AQI relate to adult asthma prevalence?  
2. What role does average household income play in state asthma rates?  
3. Are CO₂ emissions significant predictors of asthma?
""")

# ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(
        df, x='MedianAQI', y='AsthmaPrev',
        size='MedianAQI', color='State',
        title="Asthma Prevalence vs. Median AQI",
        labels={'MedianAQI':'Median AQI','AsthmaPrev':'Asthma Prevalence (%)'}
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Interpretation:** No clear upward trend—states with higher median AQI do not systematically exhibit higher asthma rates (p > 0.6).")

with col2:
    fig2 = px.scatter(
        df, x='AvgIncome', y='AsthmaPrev',
        color='AvgIncome', trendline='ols',
        title="Asthma Prevalence vs. Avg Household Income",
        labels={'AvgIncome':'Avg Income (USD)','AsthmaPrev':'Asthma Prevalence (%)'},
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Interpretation:** Income shows no significant linear effect on asthma at the state scale (p > 0.4).")

st.write("---")

st.header("CO₂ Emissions Analysis")

fig3 = px.bar(
    df.sort_values('CarbonPerCapita2021', ascending=False),
    x='State', y='CarbonPerCapita2021',
    title="Per‑Capita CO₂ Emissions by State (2021)",
    labels={'CarbonPerCapita2021':'CO₂ per Capita (metric tons)'}
)
fig3.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig3, use_container_width=True)
st.markdown("**Interpretation:** Wide variation in CO₂, but no strong link to asthma rates (p > 0.2).")

st.write("---")

st.header("🔍 Final Conclusion")
st.markdown("""
Replacing “days with AQI” with **median AQI** did not improve our models—median AQI, income, and CO₂ remain non‑significant predictors of state asthma prevalence.  
• **R² values** stay near zero, meaning these state‑level linear models explain almost none of the variation.  
**Next steps:**  
- Move to county‑level or panel data for finer resolution  
- Enrich with health/demographic covariates (smoking, age, healthcare access)  
- Apply regularized or tree‑based models to capture non‑linear effects  
""")

st.markdown("""
---
**INFOSCI 301 | Duke Kunshan University, 2025**  
*Authors: Nazirjon Ismoiljonov & Jiean Zhou*
""")
