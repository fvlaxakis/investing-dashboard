-- ============================================================================
-- Supabase setup για το sync του χαρτοφυλακίου.
-- Τρέξε το ΜΙΑ φορά στο Supabase → SQL Editor → New query → Run.
-- ============================================================================

create table if not exists portfolio (
  id        bigint generated always as identity primary key,
  symbol    text not null,
  qty       numeric,
  buy_price numeric,
  created_at timestamptz default now()
);

-- Ενεργοποίηση Row Level Security
alter table portfolio enable row level security;

-- Πολιτική: επιτρέπει στο anon key (που θα μπει στα Streamlit secrets) πλήρη
-- πρόσβαση. Το κλειδί ΔΕΝ είναι δημόσιο — ζει στα server-side secrets του
-- Streamlit, όχι στη σελίδα. Είναι προσωπική εφαρμογή ενός χρήστη.
drop policy if exists "app full access" on portfolio;
create policy "app full access" on portfolio
  for all
  to anon
  using (true)
  with check (true);
