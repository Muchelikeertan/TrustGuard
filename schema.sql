-- ==============================================================================
-- TrustGuard PostgreSQL Schema for Supabase
-- Project PS-02 | Innovators Conclave 2026
-- Tagline: "AI for Digital Trust — Multimodal Identity Consistency Platform"
-- ==============================================================================

-- 1. Profiles Table (linked to Supabase auth.users)
create table if not exists public.profiles (
    id uuid references auth.users(id) on delete cascade primary key,
    email text,
    full_name text,
    organization text,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Enable Row Level Security (RLS) on profiles
alter table public.profiles enable row level security;

-- Profiles Policies
create policy "Users can view their own profile"
    on public.profiles for select
    using (auth.uid() = id);

create policy "Users can insert their own profile"
    on public.profiles for insert
    with check (auth.uid() = id);

create policy "Users can update their own profile"
    on public.profiles for update
    using (auth.uid() = id);

-- 2. Analysis Sessions Table
create table if not exists public.analysis_sessions (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade not null,
    title text not null,
    claimed_identity text not null,
    chat_text text default '',
    risk_level text not null check (risk_level in ('LOW', 'UNCERTAIN', 'HIGH')),
    modality_statuses jsonb not null default '{}'::jsonb,
    contradiction_map jsonb not null default '{"agreements":[], "conflicts":[], "uncertain":[]}'::jsonb,
    explanation text not null,
    recommended_action text not null,
    details jsonb not null default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Query Indexes
create index if not exists idx_analysis_sessions_user_id on public.analysis_sessions(user_id);
create index if not exists idx_analysis_sessions_created_at on public.analysis_sessions(created_at desc);

-- Enable Row Level Security (RLS) on analysis_sessions
alter table public.analysis_sessions enable row level security;

-- Analysis Sessions Policies (Strict Isolation per user)
create policy "Users can view their own analysis sessions"
    on public.analysis_sessions for select
    using (auth.uid() = user_id);

create policy "Users can insert their own analysis sessions"
    on public.analysis_sessions for insert
    with check (auth.uid() = user_id);

create policy "Users can update their own analysis sessions"
    on public.analysis_sessions for update
    using (auth.uid() = user_id);

create policy "Users can delete their own analysis sessions"
    on public.analysis_sessions for delete
    using (auth.uid() = user_id);

-- Optional: Automatic updated_at timestamp trigger
create or replace function public.handle_updated_at()
returns trigger as $$
begin
    new.updated_at = timezone('utc'::text, now());
    return new;
end;
$$ language plpgsql;

create or replace trigger set_profiles_updated_at
    before update on public.profiles
    for each row execute function public.handle_updated_at();

create or replace trigger set_analysis_sessions_updated_at
    before update on public.analysis_sessions
    for each row execute function public.handle_updated_at();
