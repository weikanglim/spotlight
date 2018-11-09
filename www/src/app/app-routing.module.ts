import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { TextSearchComponent } from './text-search/text-search.component';
import { ClassifierComponent } from './classifier/classifier.component';
import { HomeComponent } from './home/home.component';
import { DataComponent } from './data/data.component';
import { DataAddComponent } from './data-add/data-add.component';

const routes: Routes = [
  { path : '', component: HomeComponent },
  { path : 'data', component: DataComponent },
  { path : 'add-data', component: DataAddComponent },
  { path : 'text-search', component: TextSearchComponent },
  { path : 'classifier', component: ClassifierComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes, { enableTracing : true })],
  exports: [RouterModule]
})
export class AppRoutingModule { }
